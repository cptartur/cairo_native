use std::collections::BTreeMap;

use itertools::chain;
use starknet_types_core::felt::Felt;

use cairo_lang_sierra::{extensions::gas::CostTokenType, ids::FunctionId, program::Program};
use cairo_lang_sierra_to_casm::metadata::MetadataComputationConfig;
use cairo_lang_starknet_classes::casm_contract_class::ENTRY_POINT_COST;
use cairo_lang_starknet_classes::compiler_version::VersionId;
use cairo_lang_starknet_classes::contract_class::ContractEntryPoints;

use crate::{
    context::NativeContext, error::Result, executor::jit::JitNativeExecutor,
    statistics::Statistics, OptLevel,
};

use super::{find_entrypoint_builtins, ContractExecutor, EntryPointInfo, NativeContractInfo};

/// JIT-based contract executor that mirrors AOT behavior but keeps everything in-memory.
#[derive(Debug)]
pub struct JitContractExecutor {
    executor: JitNativeExecutor<'static>,
    contract_info: NativeContractInfo,
}

impl ContractExecutor for JitContractExecutor {
    fn new(
        program: &Program,
        entry_points: &ContractEntryPoints,
        sierra_version: VersionId,
        opt_level: OptLevel,
        stats: Option<&mut Statistics>,
    ) -> Result<Self> {
        // Configure linear solvers based on Sierra version, same as AOT.
        let no_eq_solver = match sierra_version.major.cmp(&1) {
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => sierra_version.minor >= 4,
            std::cmp::Ordering::Greater => true,
        };

        let leaked_ctx: &'static NativeContext = Box::leak(Box::new(NativeContext::new()));

        // Compile the Sierra program to a NativeModule with contract gas metadata.
        let native_module = leaked_ctx.compile(
            program,
            true,
            Some(MetadataComputationConfig {
                function_set_costs: chain!(
                    entry_points.constructor.iter(),
                    entry_points.external.iter(),
                    entry_points.l1_handler.iter(),
                )
                .map(|x| {
                    (
                        FunctionId::new(x.function_idx as u64),
                        [(CostTokenType::Const, ENTRY_POINT_COST)].into(),
                    )
                })
                .collect(),
                linear_gas_solver: no_eq_solver,
                linear_ap_change_solver: no_eq_solver,
                skip_non_linear_solver_comparisons: false,
                compute_runtime_costs: false,
            }),
            crate::clone_option_mut!(stats),
        )?;

        // Build JIT executor from the compiled module.
        let executor = JitNativeExecutor::from_native_module(native_module, opt_level)?;

        // Build the selector -> function mapping and builtin layout information (same as AOT).
        let registry = executor.program_registry();
        let entry_point_mappings: BTreeMap<Felt, EntryPointInfo> = chain!(
            entry_points.constructor.iter(),
            entry_points.external.iter(),
            entry_points.l1_handler.iter(),
        )
        .map(|x| {
            let function_id = x.function_idx as u64;
            let function = registry
                .get_function(&FunctionId::new(function_id))
                .expect("function must exist in registry");

            let builtins = find_entrypoint_builtins(function, registry)?;

            Ok((
                Felt::from(&x.selector),
                EntryPointInfo {
                    function_id,
                    builtins,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self {
            executor,
            contract_info: NativeContractInfo {
                version: super::ContractInfoVersion::V0,
                entry_points: entry_point_mappings,
            },
        })
    }

    // For JIT, there is no on-disk artifact. Build in-memory and return Some.
    fn new_into(
        program: &Program,
        entry_points: &ContractEntryPoints,
        sierra_version: VersionId,
        _output_path: impl Into<std::path::PathBuf>,
        opt_level: OptLevel,
        stats: Option<&mut Statistics>,
    ) -> Result<Option<Self>> {
        Self::new(program, entry_points, sierra_version, opt_level, stats).map(Some)
    }

    // JIT executors cannot be loaded from disk; return None to signal to caller.
    fn from_path(_path: impl Into<std::path::PathBuf>) -> Result<Option<Self>> {
        Ok(None)
    }

    fn run<H: crate::starknet::StarknetSyscallHandler>(
        &self,
        selector: Felt,
        args: &[Felt],
        gas: u64,
        builtin_costs: Option<crate::utils::BuiltinCosts>,
        syscall_handler: H,
    ) -> Result<crate::execution_result::ContractExecutionResult> {
        use crate::executor::BuiltinCostsGuard;

        // Resolve selector to function id.
        let entry = self
            .contract_info
            .entry_points
            .get(&selector)
            .ok_or(crate::error::Error::SelectorNotFound)?;
        let function_id = FunctionId::new(entry.function_id);

        // Install builtin costs for the duration of the call (matches AOT behavior).
        let builtin_costs_guard = BuiltinCostsGuard::install(builtin_costs.unwrap_or_default());

        let result = self.executor.invoke_contract_dynamic(
            &function_id,
            args,
            Some(gas),
            syscall_handler,
        )?;

        drop(builtin_costs_guard);

        Ok(result)
    }
}
