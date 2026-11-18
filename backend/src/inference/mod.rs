pub mod candle_sam2;
pub mod candle_sam3;
pub mod candle_sam3_frame_source;
pub mod candle_sam3_helpers;
pub mod image;
pub mod registry;
pub mod video;

/// Raw model masks and the preprocessing decisions that produced them.
pub struct SegmentationOutput {
    pub masks: Vec<image::FrameMask>,
    pub input_window: Option<crate::imaging::input_window::InputWindowReport>,
}
