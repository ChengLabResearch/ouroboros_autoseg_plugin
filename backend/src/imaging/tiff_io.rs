use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{error::AppError, imaging::annotations::AnnotationPoint};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageFrame {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeGeometry {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone)]
pub struct VolumeInput {
    pub source: PathBuf,
    pub geometry: VolumeGeometry,
    pub annotation_points: Option<Vec<AnnotationPoint>>,
}

#[derive(Debug, Clone)]
pub(crate) struct WritableTiffPage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedTiff {
    bytes: Vec<u8>,
    endian: Endian,
    pages: Vec<ParsedPage>,
}

#[derive(Debug, Clone)]
struct ParsedPage {
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    bits_per_sample: Vec<u16>,
    compression: u16,
    photometric_interpretation: u16,
    strip_offsets: Vec<u32>,
    strip_byte_counts: Vec<u32>,
    rows_per_strip: usize,
    planar_configuration: u16,
    description: Option<String>,
}

#[derive(Debug, Clone)]
struct IfdEntry {
    tag: u16,
    type_id: u16,
    count: u32,
    value_field: [u8; 4],
}

#[derive(Debug, Clone)]
struct PageLayout {
    pixel_offset: u32,
    bits_offset: Option<u32>,
    description_offset: Option<u32>,
    description_bytes: Option<Vec<u8>>,
    ifd_offset: u32,
    next_ifd_offset: u32,
    entry_count: u16,
}

#[derive(Debug, Clone, Copy)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
            Self::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
        }
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

pub async fn inspect_volume(path: &Path) -> Result<VolumeInput, AppError> {
    inspect_volume_sync(path)
}

pub async fn read_volume_frames(path: &Path) -> Result<Vec<ImageFrame>, AppError> {
    read_volume_frames_sync(path)
}

pub(crate) fn write_tiff_pages(path: &Path, pages: &[WritableTiffPage]) -> Result<(), AppError> {
    if pages.is_empty() {
        return Err(AppError::bad_request(
            "TIFF output must contain at least one page",
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    for page in pages {
        validate_writable_page(page)?;
    }

    let mut cursor = 8u32;
    let mut layouts = Vec::with_capacity(pages.len());
    for page in pages {
        let pixel_offset = cursor;
        cursor = cursor
            .checked_add(page.pixels.len() as u32)
            .ok_or_else(|| AppError::bad_request("TIFF output is too large"))?;

        let bits_offset = if page.channels == 3 {
            let offset = cursor;
            cursor = cursor
                .checked_add(6)
                .ok_or_else(|| AppError::bad_request("TIFF output is too large"))?;
            Some(offset)
        } else {
            None
        };

        let description_bytes = page.description.as_ref().map(|value| ascii_with_nul(value));
        let description_offset = if let Some(bytes) = &description_bytes {
            let offset = cursor;
            cursor = cursor
                .checked_add(bytes.len() as u32)
                .ok_or_else(|| AppError::bad_request("TIFF output is too large"))?;
            Some(offset)
        } else {
            None
        };

        layouts.push(PageLayout {
            pixel_offset,
            bits_offset,
            description_offset,
            description_bytes,
            ifd_offset: 0,
            next_ifd_offset: 0,
            entry_count: 0,
        });
    }

    for (index, page) in pages.iter().enumerate() {
        let entry_count = 9
            + usize::from(page.channels == 3)
            + usize::from(layouts[index].description_bytes.is_some());
        layouts[index].entry_count = entry_count as u16;
        layouts[index].ifd_offset = cursor;
        cursor = cursor
            .checked_add(2 + entry_count as u32 * 12 + 4)
            .ok_or_else(|| AppError::bad_request("TIFF output is too large"))?;
    }

    for index in 0..layouts.len().saturating_sub(1) {
        layouts[index].next_ifd_offset = layouts[index + 1].ifd_offset;
    }

    let mut file = fs::File::create(path)?;
    file.write_all(b"II")?;
    file.write_all(&42u16.to_le_bytes())?;
    file.write_all(&layouts[0].ifd_offset.to_le_bytes())?;

    for page in pages {
        file.write_all(&page.pixels)?;
        if page.channels == 3 {
            for _ in 0..3 {
                file.write_all(&8u16.to_le_bytes())?;
            }
        }
        if let Some(description) = &page.description {
            file.write_all(&ascii_with_nul(description))?;
        }
    }

    for (page, layout) in pages.iter().zip(&layouts) {
        file.write_all(&layout.entry_count.to_le_bytes())?;
        write_ifd_entry(&mut file, 256, 4, 1, (page.width as u32).to_le_bytes())?;
        write_ifd_entry(&mut file, 257, 4, 1, (page.height as u32).to_le_bytes())?;
        let bits_value = if let Some(offset) = layout.bits_offset {
            offset.to_le_bytes()
        } else {
            inline_short(8)
        };
        write_ifd_entry(
            &mut file,
            258,
            3,
            if page.channels == 3 { 3 } else { 1 },
            bits_value,
        )?;
        write_ifd_entry(&mut file, 259, 3, 1, inline_short(1))?;
        write_ifd_entry(
            &mut file,
            262,
            3,
            1,
            inline_short(if page.channels == 3 { 2 } else { 1 }),
        )?;
        if let Some(description_offset) = layout.description_offset {
            write_ifd_entry(
                &mut file,
                270,
                2,
                layout
                    .description_bytes
                    .as_ref()
                    .map(|value| value.len() as u32)
                    .unwrap_or_default(),
                description_offset.to_le_bytes(),
            )?;
        }
        write_ifd_entry(&mut file, 273, 4, 1, layout.pixel_offset.to_le_bytes())?;
        write_ifd_entry(&mut file, 277, 3, 1, inline_short(page.channels as u16))?;
        write_ifd_entry(&mut file, 278, 4, 1, (page.height as u32).to_le_bytes())?;
        write_ifd_entry(
            &mut file,
            279,
            4,
            1,
            (page.pixels.len() as u32).to_le_bytes(),
        )?;
        if page.channels == 3 {
            write_ifd_entry(&mut file, 284, 3, 1, inline_short(1))?;
        }
        file.write_all(&layout.next_ifd_offset.to_le_bytes())?;
    }

    Ok(())
}

fn inspect_volume_sync(path: &Path) -> Result<VolumeInput, AppError> {
    if path.is_dir() {
        let frames = sorted_tiff_paths(path)?;
        let first_frame = frames.first().ok_or_else(|| {
            AppError::bad_request("Input directory does not contain any TIFF files")
        })?;
        let parsed = read_tiff_file(first_frame)?;
        let first_page = parsed
            .pages
            .first()
            .ok_or_else(|| invalid_tiff("input directory contains an empty TIFF"))?;

        Ok(VolumeInput {
            source: path.to_path_buf(),
            geometry: VolumeGeometry {
                frames: frames.len(),
                height: first_page.height,
                width: first_page.width,
            },
            annotation_points: parse_annotation_points(first_page.description.as_deref()),
        })
    } else {
        let parsed = read_tiff_file(path)?;
        let first_page = parsed
            .pages
            .first()
            .ok_or_else(|| invalid_tiff("TIFF file does not contain any pages"))?;

        Ok(VolumeInput {
            source: path.to_path_buf(),
            geometry: VolumeGeometry {
                frames: parsed.pages.len(),
                height: first_page.height,
                width: first_page.width,
            },
            annotation_points: parse_annotation_points(first_page.description.as_deref()),
        })
    }
}

fn read_volume_frames_sync(path: &Path) -> Result<Vec<ImageFrame>, AppError> {
    if path.is_dir() {
        let mut frames = Vec::new();
        for frame_path in sorted_tiff_paths(path)? {
            let parsed = read_tiff_file(&frame_path)?;
            let first_page = parsed
                .pages
                .first()
                .ok_or_else(|| invalid_tiff("input directory contains an empty TIFF"))?;
            frames.push(decode_page_pixels(&parsed, first_page)?);
        }
        if frames.is_empty() {
            return Err(AppError::bad_request(
                "Input directory does not contain any TIFF files",
            ));
        }
        Ok(frames)
    } else {
        let parsed = read_tiff_file(path)?;
        parsed
            .pages
            .iter()
            .map(|page| decode_page_pixels(&parsed, page))
            .collect()
    }
}

fn sorted_tiff_paths(folder: &Path) -> Result<Vec<PathBuf>, AppError> {
    let mut paths = fs::read_dir(folder)?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| matches_tiff_extension(path))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn matches_tiff_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase()),
        Some(ref value) if value == "tif" || value == "tiff"
    )
}

fn parse_annotation_points(description: Option<&str>) -> Option<Vec<AnnotationPoint>> {
    let description = description?;
    let metadata: Value = serde_json::from_str(description).ok()?;
    let rows = metadata.get("annotation_points")?.as_array()?;
    if rows.is_empty() {
        return None;
    }

    let mut points = Vec::with_capacity(rows.len());
    for row in rows {
        let values = row.as_array()?;
        if values.len() < 3 {
            return None;
        }
        points.push(AnnotationPoint {
            x: values[0].as_f64()? as f32,
            y: values[1].as_f64()? as f32,
            z: values[2].as_f64()? as f32,
        });
    }

    Some(points)
}

fn read_tiff_file(path: &Path) -> Result<ParsedTiff, AppError> {
    parse_tiff_bytes(fs::read(path)?)
}

fn parse_tiff_bytes(bytes: Vec<u8>) -> Result<ParsedTiff, AppError> {
    if bytes.len() < 8 {
        return Err(invalid_tiff("file is too small to be a TIFF"));
    }

    let endian = match &bytes[..2] {
        b"II" => Endian::Little,
        b"MM" => Endian::Big,
        _ => return Err(invalid_tiff("missing TIFF byte-order marker")),
    };
    if endian.read_u16(&bytes[2..4]) != 42 {
        return Err(invalid_tiff("unsupported TIFF magic value"));
    }

    let mut pages = Vec::new();
    let mut next_ifd_offset = endian.read_u32(&bytes[4..8]) as usize;
    let mut seen_offsets = HashSet::new();
    while next_ifd_offset != 0 {
        if !seen_offsets.insert(next_ifd_offset) {
            return Err(invalid_tiff("encountered an IFD loop"));
        }
        let (page, following_ifd_offset) = parse_ifd(&bytes, next_ifd_offset, endian)?;
        pages.push(page);
        next_ifd_offset = following_ifd_offset;
    }

    if pages.is_empty() {
        return Err(invalid_tiff("TIFF file does not contain any pages"));
    }

    Ok(ParsedTiff {
        bytes,
        endian,
        pages,
    })
}

fn parse_ifd(
    bytes: &[u8],
    ifd_offset: usize,
    endian: Endian,
) -> Result<(ParsedPage, usize), AppError> {
    let entry_count = endian.read_u16(slice(bytes, ifd_offset, 2)?) as usize;
    let entries_offset = ifd_offset + 2;
    let next_ifd_field_offset = entries_offset + entry_count * 12;
    let mut entries = Vec::with_capacity(entry_count);

    for index in 0..entry_count {
        let entry_offset = entries_offset + index * 12;
        let entry_bytes = slice(bytes, entry_offset, 12)?;
        let mut value_field = [0u8; 4];
        value_field.copy_from_slice(&entry_bytes[8..12]);
        entries.push(IfdEntry {
            tag: endian.read_u16(&entry_bytes[0..2]),
            type_id: endian.read_u16(&entry_bytes[2..4]),
            count: endian.read_u32(&entry_bytes[4..8]),
            value_field,
        });
    }

    let next_ifd_offset = endian.read_u32(slice(bytes, next_ifd_field_offset, 4)?) as usize;
    let width = entry_as_usize(bytes, required_entry(&entries, 256, "ImageWidth")?, endian)?;
    let height = entry_as_usize(bytes, required_entry(&entries, 257, "ImageLength")?, endian)?;
    let bits_per_sample = if let Some(entry) = find_entry(&entries, 258) {
        entry_as_u16_values(bytes, entry, endian)?
    } else {
        vec![8]
    };
    let compression = if let Some(entry) = find_entry(&entries, 259) {
        entry_as_u16_values(bytes, entry, endian)?
            .first()
            .copied()
            .unwrap_or(1)
    } else {
        1
    };
    let photometric_interpretation = if let Some(entry) = find_entry(&entries, 262) {
        entry_as_u16_values(bytes, entry, endian)?
            .first()
            .copied()
            .unwrap_or(1)
    } else {
        1
    };
    let strip_offsets = entry_as_u32_values(
        bytes,
        required_entry(&entries, 273, "StripOffsets")?,
        endian,
    )?;
    let samples_per_pixel = if let Some(entry) = find_entry(&entries, 277) {
        entry_as_usize(bytes, entry, endian)?
    } else {
        bits_per_sample.len().max(1)
    };
    let rows_per_strip = if let Some(entry) = find_entry(&entries, 278) {
        entry_as_usize(bytes, entry, endian)?
    } else {
        height
    };
    let strip_byte_counts = entry_as_u32_values(
        bytes,
        required_entry(&entries, 279, "StripByteCounts")?,
        endian,
    )?;
    let planar_configuration = if let Some(entry) = find_entry(&entries, 284) {
        entry_as_u16_values(bytes, entry, endian)?
            .first()
            .copied()
            .unwrap_or(1)
    } else {
        1
    };
    let description = find_entry(&entries, 270)
        .map(|entry| entry_as_ascii_string(bytes, entry, endian))
        .transpose()?;

    if strip_offsets.len() != strip_byte_counts.len() {
        return Err(invalid_tiff(
            "StripOffsets and StripByteCounts contain different numbers of entries",
        ));
    }

    Ok((
        ParsedPage {
            width,
            height,
            samples_per_pixel,
            bits_per_sample,
            compression,
            photometric_interpretation,
            strip_offsets,
            strip_byte_counts,
            rows_per_strip,
            planar_configuration,
            description,
        },
        next_ifd_offset,
    ))
}

fn decode_page_pixels(parsed: &ParsedTiff, page: &ParsedPage) -> Result<ImageFrame, AppError> {
    if page.compression != 1 {
        return Err(invalid_tiff(
            "only uncompressed TIFF inputs are currently supported",
        ));
    }
    if page.planar_configuration != 1 {
        return Err(invalid_tiff(
            "only chunky TIFF inputs are currently supported",
        ));
    }
    if page.samples_per_pixel == 0 {
        return Err(invalid_tiff("SamplesPerPixel must be at least one"));
    }
    if page.rows_per_strip == 0 {
        return Err(invalid_tiff("RowsPerStrip must be at least one"));
    }
    if page.photometric_interpretation != 1 && page.photometric_interpretation != 2 {
        return Err(invalid_tiff(
            "only BlackIsZero grayscale and RGB TIFF inputs are currently supported",
        ));
    }

    let bit_depth = page.bits_per_sample.first().copied().unwrap_or(8);
    if !page.bits_per_sample.iter().all(|value| *value == bit_depth) {
        return Err(invalid_tiff("mixed BitsPerSample values are not supported"));
    }
    if bit_depth != 8 && bit_depth != 16 {
        return Err(invalid_tiff(
            "only 8-bit and 16-bit TIFF inputs are supported",
        ));
    }

    let mut raw = Vec::new();
    for (&offset, &count) in page.strip_offsets.iter().zip(&page.strip_byte_counts) {
        raw.extend_from_slice(slice(&parsed.bytes, offset as usize, count as usize)?);
    }

    let expected_samples = page
        .width
        .checked_mul(page.height)
        .and_then(|value| value.checked_mul(page.samples_per_pixel))
        .ok_or_else(|| AppError::bad_request("TIFF page dimensions are too large"))?;
    let bytes_per_sample = usize::from(bit_depth / 8);
    let expected_len = expected_samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| AppError::bad_request("TIFF page dimensions are too large"))?;
    if raw.len() < expected_len {
        return Err(invalid_tiff(
            "strip data is shorter than the expected image payload",
        ));
    }

    let source = &raw[..expected_len];
    let pixels = match bit_depth {
        8 => source.to_vec(),
        16 => source
            .chunks_exact(2)
            .map(|chunk| ((parsed.endian.read_u16(chunk) / 255).min(255)) as u8)
            .collect(),
        _ => unreachable!(),
    };

    Ok(ImageFrame {
        width: page.width,
        height: page.height,
        channels: page.samples_per_pixel,
        pixels,
    })
}

fn validate_writable_page(page: &WritableTiffPage) -> Result<(), AppError> {
    if page.width == 0 || page.height == 0 {
        return Err(AppError::bad_request(
            "TIFF output pages must have non-zero dimensions",
        ));
    }
    if page.channels != 1 && page.channels != 3 {
        return Err(AppError::bad_request(
            "TIFF output currently supports only grayscale and RGB pages",
        ));
    }
    let expected_len = page
        .width
        .checked_mul(page.height)
        .and_then(|value| value.checked_mul(page.channels))
        .ok_or_else(|| AppError::bad_request("TIFF output page dimensions are too large"))?;
    if page.pixels.len() != expected_len {
        return Err(AppError::bad_request(
            "TIFF output page pixels do not match the declared dimensions",
        ));
    }
    Ok(())
}

fn write_ifd_entry<W: Write>(
    writer: &mut W,
    tag: u16,
    type_id: u16,
    count: u32,
    value_field: [u8; 4],
) -> Result<(), AppError> {
    writer.write_all(&tag.to_le_bytes())?;
    writer.write_all(&type_id.to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    writer.write_all(&value_field)?;
    Ok(())
}

fn inline_short(value: u16) -> [u8; 4] {
    let mut field = [0u8; 4];
    field[..2].copy_from_slice(&value.to_le_bytes());
    field
}

fn ascii_with_nul(value: &str) -> Vec<u8> {
    let mut bytes = value
        .as_bytes()
        .iter()
        .copied()
        .filter(|byte| *byte != 0)
        .collect::<Vec<_>>();
    bytes.push(0);
    bytes
}

fn required_entry<'a>(
    entries: &'a [IfdEntry],
    tag: u16,
    name: &str,
) -> Result<&'a IfdEntry, AppError> {
    find_entry(entries, tag).ok_or_else(|| invalid_tiff(format!("missing {name} tag")))
}

fn find_entry(entries: &[IfdEntry], tag: u16) -> Option<&IfdEntry> {
    entries.iter().find(|entry| entry.tag == tag)
}

fn entry_as_usize(bytes: &[u8], entry: &IfdEntry, endian: Endian) -> Result<usize, AppError> {
    entry_as_u32_values(bytes, entry, endian)?
        .first()
        .copied()
        .map(|value| value as usize)
        .ok_or_else(|| invalid_tiff("IFD entry did not contain any values"))
}

fn entry_as_ascii_string(
    bytes: &[u8],
    entry: &IfdEntry,
    endian: Endian,
) -> Result<String, AppError> {
    let data = entry_data(bytes, entry, endian)?;
    let trimmed = data.split(|byte| *byte == 0).next().unwrap_or(&data);
    String::from_utf8(trimmed.to_vec())
        .map_err(|_| invalid_tiff("ImageDescription metadata is not valid UTF-8"))
}

fn entry_as_u16_values(
    bytes: &[u8],
    entry: &IfdEntry,
    endian: Endian,
) -> Result<Vec<u16>, AppError> {
    let data = entry_data(bytes, entry, endian)?;
    match entry.type_id {
        1 | 7 => Ok(data.into_iter().map(u16::from).collect()),
        3 => Ok(data
            .chunks_exact(2)
            .map(|chunk| endian.read_u16(chunk))
            .collect()),
        4 => Ok(data
            .chunks_exact(4)
            .map(|chunk| endian.read_u32(chunk) as u16)
            .collect()),
        _ => Err(invalid_tiff(format!(
            "unsupported TIFF field type {} for SHORT values",
            entry.type_id
        ))),
    }
}

fn entry_as_u32_values(
    bytes: &[u8],
    entry: &IfdEntry,
    endian: Endian,
) -> Result<Vec<u32>, AppError> {
    let data = entry_data(bytes, entry, endian)?;
    match entry.type_id {
        1 | 7 => Ok(data.into_iter().map(u32::from).collect()),
        3 => Ok(data
            .chunks_exact(2)
            .map(|chunk| u32::from(endian.read_u16(chunk)))
            .collect()),
        4 => Ok(data
            .chunks_exact(4)
            .map(|chunk| endian.read_u32(chunk))
            .collect()),
        _ => Err(invalid_tiff(format!(
            "unsupported TIFF field type {} for LONG values",
            entry.type_id
        ))),
    }
}

fn entry_data(bytes: &[u8], entry: &IfdEntry, endian: Endian) -> Result<Vec<u8>, AppError> {
    let value_size = type_size(entry.type_id)?;
    let total_size = value_size
        .checked_mul(entry.count as usize)
        .ok_or_else(|| invalid_tiff("IFD entry length overflowed"))?;
    if total_size <= 4 {
        Ok(entry.value_field[..total_size].to_vec())
    } else {
        let value_offset = endian.read_u32(&entry.value_field) as usize;
        Ok(slice(bytes, value_offset, total_size)?.to_vec())
    }
}

fn type_size(type_id: u16) -> Result<usize, AppError> {
    match type_id {
        1 | 2 | 7 => Ok(1),
        3 => Ok(2),
        4 => Ok(4),
        _ => Err(invalid_tiff(format!(
            "unsupported TIFF field type {type_id}"
        ))),
    }
}

fn slice(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8], AppError> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid_tiff("TIFF offset overflowed"))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| invalid_tiff("TIFF offset is outside the file bounds"))
}

fn invalid_tiff(message: impl Into<String>) -> AppError {
    AppError::bad_request(format!("Unsupported TIFF input: {}", message.into()))
}

#[cfg(test)]
mod tests;
