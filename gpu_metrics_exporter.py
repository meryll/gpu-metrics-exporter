import logging
import prometheus_client as pc
from py3nvml.py3nvml import *
import py3nvml.nvidia_smi as smi

logger = logging.getLogger(__name__)

NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_CLOCK_VIDEO = 3
NVML_CLOCK_COUNT = 4

class GPUMetrics:

    def __init__(self):
        self._initialized = False

        try:
            smi.nvmlInit()
            self._initialized = True
            logger.debug("Successfully initialized NVIDIA SMI")
        except Exception as e:
            logger.error("Couldn't execute  metrics with NVIDIA SMI: {}".format(str(e)))
            logger.exception(e)

        if self._initialized:
            self.temperature = pc.Gauge('GPU_Temperature', 'Temperature of GPU.', labelnames=['card_id', 'card_name'])
            self.fan_speed = pc.Gauge('GPU_fan_speed', 'Fan Speed of GPU.', labelnames=['card_id', 'card_name'])
            self.memory_usage = pc.Gauge('GPU_memory_usage', 'Memory usage of GPU.', labelnames=['card_id', 'card_name', 'type'])
            self.power_usage = pc.Gauge('GPU_power_usage', 'Power usage of GPU.', labelnames=['card_id', 'card_name'])
            self.power_state = pc.Gauge('GPU_power_state', 'Power state of GPU.', labelnames=['card_id', 'card_name'])
            self.graphics_clock = pc.Gauge('GPU_graphics_clock', 'Graphics clock of GPU.', labelnames=['card_id', 'card_name'])
            self.memory_clock = pc.Gauge('GPU_memory_clock', 'Memory clock of GPU.', labelnames=['card_id', 'card_name'])
            self.power_draw = pc.Gauge('GPU_power_draw', 'Power draw of GPU.', labelnames=['card_id', 'card_name'])
            self.bar1_memory = pc.Gauge('GPU_bar1_memory', 'BAR1 memory of GPU.', labelnames=['card_id', 'card_name', 'type'])
            self.gpu_util = pc.Gauge('GPU_util', 'Utilization of GPU.', labelnames=['card_id', 'card_name', 'type'])

    def execute(self):
        if not self._initialized:
            return

        try:
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                gpu_name = nvmlDeviceGetName(handle)
                gpu_id = nvmlDeviceGetIndex(handle)

                self.temperature.labels(gpu_id, gpu_name).set(nvmlDeviceGetTemperature(handle, sensor=0))
                self.fan_speed.labels(gpu_id, gpu_name).set(nvmlDeviceGetFanSpeed(handle))
                self.power_state.labels(gpu_id, gpu_name).set(nvmlDeviceGetPowerState(handle))
                self.power_usage.labels(gpu_id, gpu_name).set(nvmlDeviceGetPowerUsage(handle))

                self.graphics_clock.labels(gpu_id, gpu_name).set(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS))
                self.memory_clock.labels(gpu_id, gpu_name).set(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM))

                memory_info = nvmlDeviceGetMemoryInfo(handle)
                self.memory_usage.labels(gpu_id, gpu_name, 'total').set(memory_info.total)
                self.memory_usage.labels(gpu_id, gpu_name, 'free').set(memory_info.free)
                self.memory_usage.labels(gpu_id, gpu_name, 'used').set(memory_info.used)

                bar1 = nvmlDeviceGetBAR1MemoryInfo(handle)
                self.bar1_memory.labels(gpu_id, gpu_name, 'bar1Total').set(bar1.bar1Total)
                self.bar1_memory.labels(gpu_id, gpu_name, 'bar1Free').set(bar1.bar1Free)
                self.bar1_memory.labels(gpu_id, gpu_name, 'bar1Used').set(bar1.bar1Used)

                util = smi.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_util.labels(gpu_id, gpu_name, 'gpu').set(util.gpu)
                self.gpu_util.labels(gpu_id, gpu_name, 'memory').set(util.memory)


        except Exception as e:
            logger.error("Couldn't update GPU metrics with NVIDIA SMI: {}".format(str(e)))
            logger.exception(e)

