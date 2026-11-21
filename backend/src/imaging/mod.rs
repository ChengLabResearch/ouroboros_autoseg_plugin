pub mod annotations;
pub mod cleanup;
pub mod input_window;
pub mod output;
pub mod overlay;
pub mod preprocess;
pub mod tiff_io;

/// Frame identity is vector position; explicit indices exist only at output.
pub(crate) fn serialize_indexed_frames<T: serde::Serialize, S: serde::Serializer>(
    frames: &[T],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(serde::Serialize)]
    struct IndexedFrame<'a, T> {
        frame_index: usize,
        #[serde(flatten)]
        frame: &'a T,
    }
    serializer.collect_seq(
        frames
            .iter()
            .enumerate()
            .map(|(frame_index, frame)| IndexedFrame { frame_index, frame }),
    )
}
