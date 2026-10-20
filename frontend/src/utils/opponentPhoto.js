// Send the original photo bytes; the backend handles EXIF orientation.
export async function prepareOpponentPhotoForUpload(file) {
  return file;
}
