"""Short supervised reasoning targets for image-restoration tool selection.

Each registered restoration action has exactly five concise alternatives.  The
dataset builder selects one alternative with a seeded RNG so rebuilding with
the same seed produces byte-identical training data.
"""

from __future__ import annotations

THINKING_TEMPLATES_PER_ACTION = 5

RESTORATION_THINKING_TEMPLATES: dict[str, tuple[str, ...]] = {
    "real_esrgan": (
        "The image looks generally soft and lacks fine detail rather than showing one dominant weather artifact. Real-ESRGAN is a suitable first step for broad detail enhancement.",
        "The main limitation appears to be weak sharpness and low apparent resolution. I will use Real-ESRGAN to recover clearer edges and textures.",
        "There is no single strong rain, snow, or haze pattern, but the image needs overall enhancement. Real-ESRGAN is the most appropriate general restoration choice.",
        "Fine structures appear blurred and the scene would benefit from super-resolution-style refinement. I will select Real-ESRGAN for this first operation.",
        "The degradation is broad softness instead of a clearly isolated noise type. Real-ESRGAN offers a balanced way to improve detail and perceptual clarity.",
    ),
    "scunet": (
        "The image contains irregular real-world noise across otherwise recognizable structures. SCUNet is well suited to blind denoising without requiring a known noise model.",
        "Random grain and local artifacts are more noticeable than blur or weather streaks. I will use SCUNet to suppress this mixed noise.",
        "The degradation resembles unknown camera noise with spatially varying strength. SCUNet is the appropriate robust denoising choice.",
        "Edges remain present, but noisy texture obscures clean detail. SCUNet should reduce the contamination while preserving scene structure.",
        "The image needs blind real-image denoising rather than a task-specific rain or haze correction. I will select SCUNet for the first step.",
    ),
    "retinexformer_fivek": (
        "The scene is underexposed with compressed shadow detail and uneven illumination. Retinexformer trained on FiveK is a strong choice for restoring brightness naturally.",
        "Dark regions hide useful structure while the overall color remains recoverable. I will use Retinexformer-FiveK for balanced low-light enhancement.",
        "The dominant problem is insufficient illumination rather than heavy noise or weather artifacts. Retinexformer-FiveK should improve exposure and local contrast.",
        "Foreground details are difficult to see because the image is too dim. Retinexformer-FiveK is appropriate for lifting shadows without treating the scene as haze.",
        "The image needs a photography-oriented exposure correction with preserved tones. I will select Retinexformer-FiveK as the first restoration action.",
    ),
    "hvicidnet": (
        "The low-light image also shows weak and distorted color in dark areas. HVI-CIDNet is suitable because it enhances illumination while modeling color information.",
        "Shadow regions are poorly separated and the scene has an unnatural dark color cast. I will use HVI-CIDNet for low-light and color recovery.",
        "The main degradation combines underexposure with reduced chromatic clarity. HVI-CIDNet is the best targeted first step for these cues.",
        "Brightness alone is not enough because dark-region colors also need correction. HVI-CIDNet should improve both visibility and color consistency.",
        "The scene appears dim with muted hues and lost shadow detail. I will select HVI-CIDNet for a color-aware low-light enhancement.",
    ),
    "lightdiff": (
        "The image has severe and spatially uneven low illumination that may require a stronger generative enhancement. LightenDiffusion is appropriate for recovering a clearer bright image.",
        "Large dark regions contain weak detail and simple exposure adjustment may be insufficient. I will use LightenDiffusion for robust low-light restoration.",
        "The dominant artifact is challenging low light with nonuniform brightness. LightenDiffusion provides a suitable first attempt for this condition.",
        "Visibility is strongly limited by darkness across the scene. I will select LightenDiffusion to reconstruct better illuminated content.",
        "The image needs substantial illumination recovery rather than only denoising. LightenDiffusion is the most fitting low-light tool for this first step.",
    ),
    "turbo_rain": (
        "Visible rain streaks are the dominant obstruction and a fast direct deraining model is appropriate. I will use Turbo-Rain to remove them first.",
        "The scene structure is mostly intact, but rain lines reduce clarity across the image. Turbo-Rain is a suitable targeted restoration action.",
        "The degradation looks like common real-world rainfall rather than haze or snow. I will select Turbo-Rain for efficient rain removal.",
        "Rain artifacts cover important edges while the underlying exposure is still usable. Turbo-Rain should provide a focused first correction.",
        "The clearest problem is image-wide rain interference. I will apply Turbo-Rain before considering any general enhancement.",
    ),
    "s2former": (
        "Long structured rain streaks cross the scene at multiple orientations. S2Former is suitable for modeling and removing these spatially organized artifacts.",
        "The rain pattern is dense enough to require strong long-range context. I will use S2Former as the first deraining operation.",
        "Repeated directional streaks obscure both smooth regions and edges. S2Former is an appropriate transformer-based rain removal choice.",
        "The dominant degradation is structured rainfall distributed over the full image. I will select S2Former to separate streaks from scene content.",
        "Rain lines interact with background textures and need context-aware removal. S2Former should handle this pattern better than a general enhancer.",
    ),
    "idt": (
        "The image contains rain streaks at varied scales and densities. IDT is appropriate for disentangling these rain components from the background.",
        "Fine and coarse rain artifacts overlap important scene textures. I will use IDT for targeted multi-pattern deraining.",
        "The rainfall is not uniform, so a dedicated image deraining transformer is preferable. IDT is the suitable first action.",
        "Rain traces vary across local regions and require adaptive removal. I will select IDT to restore cleaner structures.",
        "The main quality loss comes from complex rain streaks rather than simple sensor noise. IDT should address this degradation directly.",
    ),
    "ridcp": (
        "The scene has a washed-out veil with reduced contrast and shifted colors, typical of real-world haze. RIDCP is suitable for prior-guided dehazing.",
        "Distant structures are obscured by atmospheric scattering while nearby edges remain visible. I will use RIDCP to recover contrast and color.",
        "The dominant issue is realistic haze rather than isolated blur or noise. RIDCP is an appropriate first dehazing step.",
        "Global visibility is reduced and the image has a pale atmospheric cast. I will select RIDCP for robust real-image dehazing.",
        "The scene needs haze removal with natural color restoration. RIDCP provides the most targeted correction for these cues.",
    ),
    "kanet": (
        "The haze density varies across the scene and affects regions differently. KA-Net is suitable for adapting the dehazing strength to local content.",
        "Uneven atmospheric veil suppresses contrast in both foreground and background. I will use KA-Net for context-aware haze removal.",
        "The image shows nonuniform haze rather than a simple global brightness problem. KA-Net is the appropriate specialized first action.",
        "Local regions require different levels of visibility recovery. I will select KA-Net to handle this spatially varying haze.",
        "Atmospheric degradation is dominant and appears content dependent. KA-Net should restore clearer structures without applying one uniform correction.",
    ),
    "turbo_snow": (
        "Snow particles visibly obstruct the scene and a fast direct desnowing model is appropriate. I will use Turbo-Snow for the first cleanup step.",
        "Scattered snowflakes cover otherwise usable image content. Turbo-Snow is a suitable targeted restoration choice.",
        "The main degradation is falling snow rather than rain or sensor noise. I will select Turbo-Snow to remove these artifacts efficiently.",
        "Snow interference is distributed across the image but the background remains recognizable. Turbo-Snow should provide a focused first correction.",
        "The clearest obstruction consists of snow particles over the scene. I will apply Turbo-Snow before any general enhancement.",
    ),
    "snowmaster": (
        "The image contains dense snow with flakes of different sizes and overlapping layers. SnowMaster is appropriate for strong multi-scale desnowing.",
        "Heavy snow obscures important textures and cannot be treated as simple noise. I will use SnowMaster to recover the underlying scene.",
        "The dominant degradation is complex accumulated visual snow across depth. SnowMaster is the suitable specialized first action.",
        "Both fine flakes and larger snow streaks reduce visibility. I will select SnowMaster for comprehensive snow removal.",
        "The snow pattern is severe and spatially diverse. SnowMaster should separate these artifacts from background details effectively.",
    ),
    "nafnet_denoise": (
        "The image shows camera-like sensor noise, especially in smooth and dark regions. NAFNet-Denoise is suitable for restoring a clean signal.",
        "Fine grain contaminates local textures without a clear weather pattern. I will use NAFNet-Denoise for targeted photographic denoising.",
        "The dominant issue resembles SIDD-style real sensor noise. NAFNet-Denoise is the appropriate first restoration action.",
        "Noise is masking subtle detail while major edges remain structurally correct. I will select NAFNet-Denoise to clean the image conservatively.",
        "The scene needs high-quality denoising rather than exposure or haze correction. NAFNet-Denoise should best address these cues.",
    ),
    "focalnet_dehaze": (
        "A broad haze veil reduces contrast over both nearby and distant regions. FocalNet-Dehaze can use wide contextual information to restore visibility.",
        "The scene has globally weakened edges and atmospheric color fading. I will use FocalNet-Dehaze for context-rich haze removal.",
        "The dominant degradation is image-wide haze that benefits from a large receptive field. FocalNet-Dehaze is the suitable first action.",
        "Atmospheric scattering affects structures across several spatial scales. I will select FocalNet-Dehaze to recover global clarity.",
        "The image needs coordinated dehazing across the whole frame rather than a local adjustment. FocalNet-Dehaze should handle this effectively.",
    ),
    "focalnet_desnow": (
        "Snow artifacts appear across wide and local regions at several scales. FocalNet-Desnow is suitable for using broad context to remove them.",
        "The scene is obscured by mixed-size snowflakes that overlap background textures. I will use FocalNet-Desnow for targeted restoration.",
        "The dominant problem is distributed snow requiring multi-scale contextual separation. FocalNet-Desnow is the appropriate first action.",
        "Snow particles interfere with both fine edges and larger structures. I will select FocalNet-Desnow to recover consistent scene detail.",
        "The image needs dedicated desnowing with a wide field of view. FocalNet-Desnow should distinguish snow from the underlying content.",
    ),
    "mb_taylorformer_dehaze": (
        "The haze affects fine details and large structures at different strengths. MB-TaylorFormer-Dehaze is suitable for multi-branch, multi-scale recovery.",
        "Atmospheric veil suppresses both local texture and global contrast. I will use MB-TaylorFormer-Dehaze to restore these scales together.",
        "The dominant degradation is complex haze with detail loss across the scene. MB-TaylorFormer-Dehaze is the appropriate first operation.",
        "Visibility recovery requires preserving small edges while correcting broad haze. I will select MB-TaylorFormer-Dehaze for this balance.",
        "The image shows multi-scale dehazing needs rather than a simple exposure problem. MB-TaylorFormer-Dehaze should provide a targeted correction.",
    ),
}


def validate_thinking_templates(expected_actions: tuple[str, ...]) -> None:
    """Validate complete one-to-one coverage of the restoration action registry."""
    expected = set(expected_actions)
    actual = set(RESTORATION_THINKING_TEMPLATES)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"thinking-template action mismatch: missing={missing}, unexpected={unexpected}")

    for action, templates in RESTORATION_THINKING_TEMPLATES.items():
        if len(templates) != THINKING_TEMPLATES_PER_ACTION:
            raise ValueError(
                f"{action}: expected {THINKING_TEMPLATES_PER_ACTION} thinking templates, found {len(templates)}"
            )
        if len(set(templates)) != len(templates):
            raise ValueError(f"{action}: thinking templates must be unique")
        for index, text in enumerate(templates):
            if not text.strip() or "<think>" in text or "</think>" in text or "<tool_call>" in text:
                raise ValueError(f"{action}: invalid thinking template at index {index}")
