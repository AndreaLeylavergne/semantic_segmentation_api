from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict 

"""
# Charger le modèle avec la configuration spécifique
model = vgg_unet(n_classes=8, input_height=128, input_width=256)  
model.load_weights("Checkpoints/vgg_unet/final_weights50.weights.h5")
print("Modèle chargé")
"""

# Créer l'application FastAPI
app = FastAPI()

""""
@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    #Lire l'image reçue
    image_stream = io.BytesIO(await file.read())
    image = Image.open(image_stream).convert("RGB")
    img_array = np.array(image)

    #Faire la prédiction
    mask = model.predict_segmentation(inp=img_array)
    
    # Convertir le masque en image et le saugarder en mémoire
    mask_image = Image.fromarray(mask.astype('uint8'))
    mask_stream = io.BytesIO()
    mask_image.save(mask_stream, format='PNG')
    mask_stream.seek(0)

    data_url = base64.b64encode(mask_stream.read()).decode('utf8')
    return JSONResponse(content={"mask": "data:image/png;base64," + data_url})
"""

@app.get("/")
def root():
    """Retourne un message de bienvenue."""
    return {"Message":  "Hello, World!"}

# Exécuter l'application FastAPI sur le port 8000

"""if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""
