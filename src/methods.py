def load_model():

    try:
        return model
    except:
        try:
            model = YOLO("model.pt")
            return model
        except:
            try:
                model_link = "https://github.com/AgabaEmbedded/Ship-Detection-in-SAR-Images/releases/download/v2.0/best-12.pt"
                #"https://github.com/AgabaEmbedded/High-Signal-Road-Detection/releases/download/v2.0/best-48hrs-77.6-mAP50.pt"

                !wget -O model.pt {model_link}
                model = YOLO("model.pt")
                return model
            except Exception as e:
                print(f"Error!! fail to load model: {e}")
                return



def on_button_click(b):
    with output:
        output.clear_output()
        link = user_input.value.strip()

        if '/file/d/' in link or 'open?id=' in link:
            if '/file/d/' in link:
                file_id = link.split('/file/d/')[1].split('/')[0]
            else:
                file_id = link.split('id=')[1].split('&')[0]

            file_url = f'https://drive.google.com/uc?id={file_id}&confirm=t'

            print("Downloading Google Drive file...")

            try:
                os.chdir(upload_folder)
                gdown.download(file_url, quiet=False)
                os.chdir('/content')
                print(f"File downloaded successfully")
            except Exception as e:
                print(f"Error downloading file: {e}")
                os.chdir('/content')

        elif '/drive/folders/' in link:
            folder_id = link.split('/folders/')[1].split('?')[0]
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            print("Downloading Google Drive folder...")

            try:
                gdown.download_folder(id=folder_id, output=upload_folder, quiet=False, use_cookies=False)
                print("Folder downloaded successfully")
            except Exception as e:
                print(f"Error downloading folder: {e}")
        else:
            print("Invalid Google Drive link. Please check the format.")
def patch_image(image_folder, patch_folder, patch_size=640, stride = 420):

    metadata_dict = {}
    if not os.path.exists(patch_folder):
        os.makedirs(patch_folder)
    if len(os.listdir(image_folder)) == 0:
        print("No images found in the folder.")
        return metadata_dict

    try:
        for image_path in os.listdir(image_folder):
            patch_id = 1

            image_name = os.path.splitext(image_path)[0]
            image_patch_folder = os.path.join(patch_folder, image_name)
            os.makedirs(image_patch_folder, exist_ok=True)

            with rasterio.open(os.path.join(image_folder, image_path)) as src:
                #image = src.read()
                transform = src.transform
                crs = src.crs

                h, w = src.height, src.width
                #_, h, w = image.shape
                count = src.count

                num_patches_h = (h - patch_size) // stride + 1
                num_patches_w = (w - patch_size) // stride + 1
                total_patches = num_patches_h * num_patches_w

                metadata_dict[image_name] = {}
                metadata_dict[image_name]["transform"] = transform
                metadata_dict[image_name]["crs"] = crs

                for i in range(0, h, stride):
                    for j in range(0, w, stride):
                        window = Window(min(j, w-patch_size), min(i, h-patch_size), patch_size, patch_size)
                        patch = src.read(window=window)
                        #patch = image[:, i:i+patch_size, j:j+patch_size].copy()


                        p_min, p_max = np.percentile(patch, 0), np.percentile(patch, 100)
                        if p_max > p_min:
                            patch = np.clip(patch, p_min, p_max)
                            patch = (patch - p_min) / (p_max - p_min) * 255

                        patch = patch.astype(np.uint8)
                        if count == 1:
                            patch = patch[0]
                        else:
                            patch = np.moveaxis(patch, 0, -1)

                        patch_image = Image.fromarray(patch).convert("RGB")
                        patch_dir = f"{image_patch_folder}/{min(j, w-patch_size)}_{min(i, h-patch_size)}.png"
                        patch_image.save(patch_dir)
                        print(f"patch {patch_id}/{total_patches} saved to {patch_dir}")

                        patch_id += 1

            print(f"Done patching: {image_name}")
            #del image
            gc.collect()
        return metadata_dict
    except Exception as e:
        print(f"Error during patching: {e}")
        return metadata_dict


def bounding_boxes_to_geojson(metadata_dict, bounding_boxes, output_geojson_path):
    """
    Convert a list of bounding boxes (xywh format) from a GeoTIFF to a GeoJSON file.
    """
    crs = metadata_dict["crs"]
    transform = metadata_dict["transform"]
    features = []
    try:
        for i, (x, y, width, height) in enumerate(bounding_boxes):

            x_min = x - (width/2)
            y_min = y - (height/2)
            x_max = x + (width/2)
            y_max = y + (height/2)

            top_left = (x_min, y_min)
            top_right = (x_max, y_min)
            bottom_right = (x_max, y_max)
            bottom_left = (x_min, y_max)

            top_left_geo = rasterio.transform.xy(transform, top_left[1], top_left[0], offset='ul')
            top_right_geo = rasterio.transform.xy(transform, top_right[1], top_right[0], offset='ul')
            bottom_right_geo = rasterio.transform.xy(transform, bottom_right[1], bottom_right[0], offset='ul')
            bottom_left_geo = rasterio.transform.xy(transform, bottom_left[1], bottom_left[0], offset='ul')

            polygon_coords = [
                top_left_geo,
                top_right_geo,
                bottom_right_geo,
                bottom_left_geo,
                top_left_geo
            ]

            polygon = Polygon([polygon_coords])
            feature = Feature(
                geometry=polygon,
                properties={"id": i}
            )
            features.append(feature)

        feature_collection = FeatureCollection(
            features,
            crs={
                "type": "name",
                "properties": {"name": f"EPSG:{crs.to_epsg()}"}
            }
        )

        with open(output_geojson_path, 'w') as f:
            dump(feature_collection, f, indent=2)
        print(f"GeoJSON file saved to {output_geojson_path}")
    except Exception as e:
        print(f"Error!! fail to save geojson {patch_folder}: {e}")
    return

def inference(model, data_path, output_path, metadata_dict, geotiff_path=None, confidence = 0.7):
    for patch_folder in os.listdir(data_path):
        image_patch_folder = os.path.join(data_path, patch_folder)
        try:
            results = model.predict(image_patch_folder, save=False, imgsz=640, conf=confidence, iou=0.25, batch=8, stream=True, verbose=True)

            compiled_boxes = []
            compiled_scores = []

            for result in results:
                result = result.cpu()
                path = result.path
                boxes = result.boxes.xywh.numpy()
                scores = result.boxes.conf.numpy()

                if len(boxes) == 0:
                    continue
                image_coordinates = os.path.splitext(os.path.basename(path))[0].split("_")
                image_coordinates = list(map(int, image_coordinates))
                j, i = image_coordinates[0], image_coordinates[1]

                adjusted_boxes = boxes.copy()
                adjusted_boxes[:, 0] += j
                adjusted_boxes[:, 1] += i

                compiled_boxes.extend(adjusted_boxes.tolist())
                compiled_scores.extend(scores.tolist())

            if compiled_boxes:
                compiled_boxes = np.array(compiled_boxes)
                compiled_boxes = compiled_boxes[cv2.dnn.NMSBoxes(compiled_boxes, compiled_scores, score_threshold=confidence, nms_threshold=0.4)]
                compiled_boxes = compiled_boxes.tolist()

            else:
                print(f"No boxes detected for {patch_folder}")
                compiled_boxes = []

            bounding_boxes_to_geojson(
                metadata_dict[patch_folder],
                compiled_boxes,
                os.path.join(output_path, patch_folder + ".geojson")
            )
        except Exception as e:
            print(f"Error!! fail to get predictions for image {patch_folder}: {e}")
    return