from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.models.models import detect_food_labels
from app.utils.fatsecret_clients import search_calorie
from app.utils.openrouter_client import get_ai_eating_tips, extract_tips_from_ai_response, clean_markdown, get_calorie_tips
from pydantic import BaseModel
import shutil
import os
import json

app = FastAPI()

# Setup CORS (ubah allow_origins sesuai domain frontend kamu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalorieRequest(BaseModel):
    gender: str
    age: int
    height: int
    weight: int
    activity: float
    goal: str
    tdee: int
    goal_calories: int
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup folder templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "..", "templates"))

# Mount static folder pointing ke templates/assets agar css bisa diakses
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "..", "static")), name="static")

# Load model functions (commented out YOLO, using Roboflow now)
# model = YOLO(os.path.join(BASE_DIR, "models", "best.pt"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def detect(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/calculate-calorie", response_class=HTMLResponse)
async def calculate_calorie(request: Request):
    return templates.TemplateResponse("calculatecalorie.html", {"request": request})

@app.get("/information", response_class=HTMLResponse)
async def information(request: Request):
    # Daftar nama makanan dari ketiga model (YOLOv8, Roboflow, best.pt)
    food_list = [
        # Daftar dari YOLOv8
        "Pizza", "Burger", "Sushi", "Nasi Goreng", "Ayam Bakar", "Rendang", "Bakso", "Kentang Goreng", "Donat", "Muffin",
        "Puding", "Tempe Goreng", "Sate", "Tahu Goreng", "Telur Dadar", "Sayur Sop", "Tumis Kangkung",

        # Daftar dari Roboflow
        "Ayam Bakar", "Ayam Goreng", "Bakso", "Capcay", "Donat", "Ikan Bakar", "Ikan Goreng", "Kentang Goreng", "Kentang Rebus",
        "Nasi Putih", "Puding", "Rendang", "Roti Tawar", "Sate", "Sayur Sop", "Tahu Goreng", "Telur Ceplok", "Telur Dadar",
        "Telur Rebus", "Tempe Goreng", "Tumis Kangkung",

        # Daftar dari best.pt
        "Rice", "Eels on Rice", "Pilaf", "Chicken-'n'-Egg on Rice", "Pork Cutlet on Rice", "Beef Curry", "Sushi", "Chicken Rice",
        "Fried Rice", "Tempura Bowl", "Bibimbap", "Toast", "Croissant", "Roll Bread", "Raisin Bread", "Chip Butty", "Hamburger",
        "Pizza", "Sandwiches", "Udon Noodle", "Tempura Udon", "Soba Noodle", "Ramen Noodle", "Beef Noodle", "Tensin Noodle",
        "Fried Noodle", "Spaghetti", "Japanese-style Pancake", "Takoyaki", "Gratin", "Sauteed Vegetables", "Croquette", "Grilled Eggplant",
        "Sauteed Spinach", "Vegetable Tempura", "Miso Soup", "Potage", "Sausage", "Oden", "Omelet", "Ganmodoki", "Jiaozi", "Stew",
        "Teriyaki Grilled Fish", "Fried Fish", "Grilled Salmon", "Salmon Meuniere", "Sashimi", "Grilled Pacific Saury", "Sukiyaki",
        "Sweet and Sour Pork", "Lightly Roasted Fish", "Steamed Egg Hotchpotch", "Tempura", "Fried Chicken", "Sirloin Cutlet",
        "Nanbanzuke", "Boiled Fish", "Seasoned Beef with Potatoes", "Hamburg Steak", "Beef Steak", "Dried Fish", "Ginger Pork Saute",
        "Spicy Chili-flavored Tofu", "Yakitori", "Cabbage Roll", "Rolled Omelet", "Egg Sunny-side Up", "Fermented Soybeans", "Cold Tofu",
        "Egg Roll", "Chilled Noodle", "Stir-fried Beef and Peppers", "Simmered Pork", "Boiled Chicken and Vegetables", "Sashimi Bowl",
        "Sushi Bowl", "Fish-shaped Pancake with Bean Jam", "Shrimp with Chili Sauce", "Roast Chicken", "Steamed Meat Dumpling",
        "Omelet with Fried Rice", "Cutlet Curry", "Spaghetti Meat Sauce", "Fried Shrimp", "Potato Salad", "Green Salad", "Macaroni Salad",
        "Japanese Tofu and Vegetable Chowder", "Pork Miso Soup", "Chinese Soup", "Beef Bowl", "Kinpira-style Sauteed Burdock", "Rice Ball",
        "Pizza Toast", "Dipping Noodles", "Hot Dog", "French Fries", "Mixed Rice", "Goya Chanpuru"
    ]

    # Mengirim data ke template untuk menampilkan search engine
    return templates.TemplateResponse("information.html", {
        "request": request,
        "food_list": food_list
    })

@app.post("/detect-food/")
async def detect_food(file: UploadFile = File(...)):
    upload_folder = os.path.join(BASE_DIR, "..", "temp")
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Panggil fungsi untuk mendeteksi label makanan dan model yang digunakan
    detections = detect_food_labels(file_path)

    results = []
    for det in detections:
        label = det["label"]
        confidence = det["confidence"]
        model = det["model"]  # Menyimpan model yang digunakan
        nutrition = search_calorie(label)

        # Mendapatkan tips AI
        tips_list = []
        if nutrition:
            ai_response = get_ai_eating_tips(label, nutrition)
            raw_tips = extract_tips_from_ai_response(ai_response)
            tips_list = [clean_markdown(tip) for tip in raw_tips]

        results.append({
            "label": label,
            "confidence": confidence,
            "model": model,  # Menyertakan model dalam hasil
            "nutrition": nutrition,
            "tips": tips_list
        })

    os.remove(file_path)
    return {"detections": results}


@app.post("/generate-calorie-tips/")
async def generate_calorie_tips(req: CalorieRequest):
    raw_tips = get_calorie_tips(
        req.gender,
        req.age,
        req.height,
        req.weight,
        req.activity,
        req.goal
    )
    cleaned_tips = [clean_markdown(t) for t in raw_tips]
    return {"tips": cleaned_tips}
