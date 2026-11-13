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
diamond markers; other frames retain the existing star markers.

Smaller gaps retain more conditioning states and use more memory. Added positive
points force their interpolated location into the mask, including off-target
tissue if the trace misses the lumen.
