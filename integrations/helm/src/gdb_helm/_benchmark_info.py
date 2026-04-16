"""Static mapping of GDB benchmark IDs to HELM adapter configuration.

Each entry specifies how HELM should handle the benchmark:
- method: HELM adapter method ("generation_multimodal" or "generation")
- max_tokens: max output tokens for the adapter
- has_images: whether the benchmark's ModelInput includes images
- image_gen: whether the model output is an image (HEIM-style)
- has_video: whether images contain video paths (MP4)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkInfo:
    method: str
    max_tokens: int = 1024
    has_images: bool = False
    image_gen: bool = False
    has_video: bool = False


BENCHMARK_INFO: dict = {
    # -- category --
    "category-1": BenchmarkInfo(method="generation_multimodal", max_tokens=256, has_images=True),
    "category-2": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),

    # -- layout: understanding --
    "layout-4": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "layout-5": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "layout-6": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "layout-7": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True),

    # -- layout: generation --
    "layout-1": BenchmarkInfo(method="generation", max_tokens=0, image_gen=True),
    "layout-2": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True),
    "layout-3": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True),
    "layout-8": BenchmarkInfo(method="generation", max_tokens=0, has_images=True, image_gen=True),

    # -- svg: understanding --
    "svg-1": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "svg-2": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "svg-3": BenchmarkInfo(method="generation", max_tokens=4096),
    "svg-4": BenchmarkInfo(method="generation", max_tokens=4096),
    "svg-5": BenchmarkInfo(method="generation", max_tokens=4096),

    # -- svg: generation --
    "svg-6": BenchmarkInfo(method="generation", max_tokens=4096),
    "svg-7": BenchmarkInfo(method="generation_multimodal", max_tokens=4096, has_images=True),
    "svg-8": BenchmarkInfo(method="generation_multimodal", max_tokens=4096, has_images=True),

    # -- template --
    "template-1": BenchmarkInfo(method="generation_multimodal", max_tokens=64, has_images=True),
    "template-2": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),
    "template-3": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),
    "template-4": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True),
    "template-5": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True),

    # -- temporal --
    "temporal-1": BenchmarkInfo(method="generation_multimodal", max_tokens=256, has_images=True),
    "temporal-2": BenchmarkInfo(method="generation_multimodal", max_tokens=128, has_images=True, has_video=True),
    "temporal-3": BenchmarkInfo(method="generation_multimodal", max_tokens=2048, has_images=True, has_video=True),
    "temporal-4": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),
    "temporal-5": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),
    "temporal-6": BenchmarkInfo(method="generation", max_tokens=512),

    # -- typography: understanding --
    "typography-1": BenchmarkInfo(method="generation_multimodal", max_tokens=128, has_images=True),
    "typography-2": BenchmarkInfo(method="generation_multimodal", max_tokens=128, has_images=True),
    "typography-3": BenchmarkInfo(method="generation_multimodal", max_tokens=512, has_images=True),
    "typography-4": BenchmarkInfo(method="generation_multimodal", max_tokens=1024, has_images=True),
    "typography-5": BenchmarkInfo(method="generation_multimodal", max_tokens=256, has_images=True),
    "typography-6": BenchmarkInfo(method="generation_multimodal", max_tokens=256, has_images=True),

    # -- typography: generation --
    "typography-7": BenchmarkInfo(method="generation", max_tokens=0, has_images=True, image_gen=True),
    "typography-8": BenchmarkInfo(method="generation", max_tokens=0, image_gen=True),

    # -- lottie --
    "lottie-1": BenchmarkInfo(method="generation", max_tokens=4096),
    "lottie-2": BenchmarkInfo(method="generation_multimodal", max_tokens=4096, has_images=True),
}
