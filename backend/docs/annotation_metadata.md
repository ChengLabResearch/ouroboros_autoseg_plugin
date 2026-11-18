# Straightened TIFF Annotation Metadata

Ouroboros slicing output may include prompt seed points in the TIFF
`ImageDescription` metadata under the `annotation_points` key. The plugin reads this
metadata from the first page of a single-stack TIFF, or from the first TIFF in a
directory-of-TIFF input.

The value must be a non-empty array of `[x, y, z]` rows:

```json
{
  "annotation_points": [
    [128.0, 96.0, 0.0],
    [127.5, 95.25, 12.0]
  ]
}
```

- `x`: pixel coordinate across the straightened slice width.
- `y`: pixel coordinate across the straightened slice height.
- `z`: frame or depth coordinate in the straightened stack.

All coordinates are expressed in straightened-volume coordinates after Ouroboros
projects the source annotation path into the sliced volume. They are not source
scan coordinates.

Missing `annotation_points` metadata means the plugin may use its configured
fallback center prompts. Present but malformed `annotation_points` metadata is a
bad request and must not silently fall back, because that would hide an invalid
Ouroboros-to-plugin handoff.

VideoPredictor can add prompts with the optional `max_prompt_gap` request field
(slices). Omitted, `null`, or `0` keeps the original prompts. A positive integer
splits gaps between consecutive prompted frames into evenly spaced intervals no
larger than that value, rounding to the nearest frame and interpolating x/y from
the original annotations. Existing annotations remain unchanged, including
multiple points that round to the same frame. The same option applies to fallback
center prompts. It has no effect on ImagePredictor.

Only internal gaps are filled: no prompts are added before the first or after the
last prompted frame. Video propagation remains forward-only. The job log reports
the added count. With **Draw Prompts** enabled, added prompt frames have hollow
diamond markers; original metadata and fallback positives have star markers.

Smaller gaps retain more conditioning states and use more memory. Added positive
points force their interpolated location into the mask, including off-target
tissue if the trace misses the lumen.

## Prompt overlay

**Draw Prompts** writes a separate annotation-overlay TIFF using the same
prepared point list passed to inference. It does not interpolate additional
display points. In video mode, frames without prepared prompts, including
leading and trailing frames, remain pixel-identical to the primary mask unless
marked as repaired by the optional inconsistent-frame repair step or outlined
by the optional input window.
Multiple annotations that round to the same frame all appear; they are not
collapsed into one point. Image mode continues to mark every actually prompted
frame using its prepared points.

The marker key is:

| Kind | Marker | Default intensity |
| --- | --- | --- |
| Metadata or fallback positive | Star | 127 |
| Gap-fill positive | Hollow diamond | 127 |
| Input window | Rectangle at native window bounds | 96 |
| Repaired frame | X centred at (2, 2) | 192 |

Point coordinates are rounded to pixels and glyphs are clipped at image
edges. The diagnostic represents submitted prompts; the runtime's 16-point
limit may omit excess points during encoding. The primary mask and the centre-line cleanup's
interpolated seeds are unchanged by drawing prompts.

Repair markers derive from the same per-frame report saved in the JSON sidecar;
they are drawn after prompt markers, including on unprompted video frames.
They use intensity 191 if the positive intensity is 192. Actions `clipped` and
`replaced` receive an X; `kept` frames do not. Recovered anchors retain their
original detector reasons in the report but leave the final flag list, while
unchanged flagged frames remain in that list without an X. Repair uses the
existing interpolated trace for component selection, independently of the sparse list
of inference prompts. See [inconsistent-frame repair](../../README.md#inconsistent-frame-repair)
for the request options, rule, and output files.

Input-window rectangles derive from the recorded preprocessing bounds in the
same sidecar. They appear on every frame when windowing is enabled, including
unprompted video frames and the full-frame automatic bootstrap. Rectangles
are drawn before point and repair markers, at intensity 95 when the positive
intensity is 96. They affect only the diagnostic TIFF. See
[input window](../../README.md#input-window) for automatic sizing, raw-mask
feedback, and the schema-version-3 sidecar fields.
