""" config/cuda_info.py """

import os
import yaml
import torch


class CudaInfo:

    def __init__(self, output_dir=""):
        self.output_path = os.path.join(output_dir, "cuda_info.yaml")
        os.makedirs(output_dir, exist_ok=True)

    def gather_info(self):
        """ Gather CUDA and system compatibility information. """
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        cpu_compatible = True if torch.has_mkl else False

        info = {
            "CUDA_Device_Information": {
                "CUDA_Available": cuda_available,
                "Number_of_CUDA_Devices": device_count,
            }
        }

        if cuda_available:
            driver_version = getattr(torch.cuda, "driver_version", None)
            info["CUDA_Device_Information"].update({
                "CUDA_Version": torch.version.cuda,
                "cuDNN_Enabled": torch.backends.cudnn.enabled,
                "Driver_Version": driver_version,
                "Current_Device": f"{torch.cuda.current_device()} ({torch.cuda.get_device_name()})",
            })

            devices_info = {}
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "Device_Name": torch.cuda.get_device_name(i),
                    "Compute_Capability": device_props.major if hasattr(device_props, 'major') else None,
                    "Total_Memory_GB": f"{device_props.total_memory / (1024**3):.2f}",
                    "Multi_Processor_Count": getattr(device_props, 'multi_processor_count', None),
                    "Max_Threads_per_Block": getattr(device_props, 'max_threads_per_block', None),
                    "Max_Thread_Dimensions": getattr(device_props, 'max_threads_dim', None),
                    "Max_Grid_Size": getattr(device_props, 'max_grid_size', None),
                    "Memory_Clock_Rate_MHz": f"{getattr(device_props, 'memory_clock_rate', 0) / 1000:.2f}",
                    "Memory_Bus_Width_bits": getattr(device_props, 'memory_bus_width', None),
                    "Warp_Size": getattr(device_props, 'warp_size', None),
                    "Allocated_Memory_GB": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f}",
                    "Cached_Memory_GB": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f}",
                    "Free_Memory_GB": f"{(device_props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3):.2f}",
                }
                devices_info[f"Device_{i}"] = device_info
            info["CUDA_Device_Information"]["Devices"] = devices_info

        info["Additional_Compatibility"] = {
            "Stream_Synchronization_Supported": str(cuda_available and hasattr(torch.cuda, 'Stream')),
            "Supports_Half_Precision": str(cuda_available and torch.cuda.get_device_capability()[0] >= 5),
            "CPU_Compatible": cpu_compatible
        }

        return info

    def save_to_yaml(self):
        """ Save gathered information to a YAML file. """
        info = self.gather_info()
        with open(self.output_path, "w") as f:
            yaml.dump(info, f, default_flow_style=False)
        print(f"CUDA and compatibility information saved to {self.output_path}\n", flush=True)
