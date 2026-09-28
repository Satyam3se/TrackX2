from open_image_models.detection.core.hub import download_model
p = download_model('yolo-v9-t-384-license-plate-end2end', force_download=True)
print('detector model:', p)