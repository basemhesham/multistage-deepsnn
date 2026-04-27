# MultiStage-DeepSNN

## Project Description
**MultiStage-DeepSNN** is a high-performance hardware implementation of a Spiking Neural Network (SNN) designed for FPGA acceleration. The project features a specialized multi-stage architecture that handles temporal neural dynamics through iterative frame-based processing. 

Key technical highlights include:
*   **Iterative Filter Mapping:** Efficient runtime remapping of memory words to filter arrays across multiple computational frames.
*   **Temporal Dynamics:** Optimized for the spiking behavior of neurons, allowing for low-latency and energy-efficient inference.
*   **Parallel RTL Design:** Built using SystemVerilog to maximize hardware concurrency and throughput.

## Target Hardware
The design is synthesized and optimized for the following Xilinx Virtex UltraScale+ FPGA:

| Parameter | Specification |
|-----------|---------------|
| **Device Family** | Virtex UltraScale+ |
| **Part Number** | XCVU11P |
| **Full Part String** | `xcvu11p-flga2577-3-e` |

## Directory Structure
*   `/rtl`: Contains the SystemVerilog source files, including the core memory mapping and filter iteration logic.
*   `/docs`: implementation reports.

---