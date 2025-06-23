import os
import requests
from duckduckgo_search import DDGS

# Yaradılacaq HTML faylını saxlamaq üçün qovluq
output_dir = "xeyalin_sayti"
os.makedirs(output_dir, exist_ok=True)

# İstifadəçinin istəyi (Burada sadəcə öz cümləni yazacaqsan)
istek = input("Sayt istəklərini yaz: ")

# Sadə NLP: açar sözlərə əsaslanaraq kontent təhlili
def parse_request(request_text):
    result = {
        "image_query": None,
        "bg_color": "white",
        "music_query": None
    }
    if "tort" in request_text:
        if "çəhrayı" in request_text or "pink" in request_text:
            result["image_query"] = "pink birthday cake"
        else:
            result["image_query"] = "birthday cake"
    if "fon rəngi" in request_text:
        if "mavi" in request_text or "blue" in request_text:
            result["bg_color"] = "lightblue"
        elif "ağ" in request_text or "white" in request_text:
            result["bg_color"] = "white"
        elif "qara" in request_text or "black" in request_text:
            result["bg_color"] = "black"
    if "musiqi" in request_text:
        if "romantik" in request_text or "romantic" in request_text:
            result["music_query"] = "romantic instrumental music"
        elif "sakit" in request_text or "calm" in request_text:
            result["music_query"] = "calm background music"
    return result

# DuckDuckGo axtarışından ilk uyğun JPG və MP3 linklərini tap
def search_file(query, filetype):
    with DDGS() as ddgs:
        results = ddgs.text(f"{query} filetype:{filetype}", safesearch="Moderate", max_results=10)
        for r in results:
            if filetype in r["href"]:
                return r["href"]
    return None

# Faylı yüklə
def download_file(url, save_path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print("Yükləmə zamanı xəta:", e)
        return False
    return False

# HTML faylını yarat
def create_html(image_path, music_path, bg_color):
    html_content = f"""
    <html>
    <head><title>Xəyalın Saytı</title></head>
    <body style="background-color:{bg_color}; text-align:center; font-family:sans-serif;">
        <h1>🌟 Xoş gəlmisən! Bu sənin xəyal saytındır! 🌟</h1>
    """
    if image_path:
        html_content += f'<img src="{image_path}" alt="Tort" width="400"><br>'
    if music_path:
        html_content += f'<audio controls autoplay loop><source src="{music_path}" type="audio/mpeg">Your browser does not support the audio element.</audio>'
    html_content += "</body></html>"

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

# İstəyin təhlili
parsed = parse_request(istek)

# Şəkil və musiqi axtar və yüklə
image_url = search_file(parsed["image_query"], "jpg") if parsed["image_query"] else None
music_url = search_file(parsed["music_query"], "mp3") if parsed["music_query"] else None

image_path = os.path.join(output_dir, "image.jpg") if image_url else None
music_path = os.path.join(output_dir, "music.mp3") if music_url else None

if image_url:
    if download_file(image_url, image_path):
        print("Şəkil uğurla yükləndi.")
if music_url:
    if download_file(music_url, music_path):
        print("Musiqi uğurla yükləndi.")

# HTML faylını yarat
create_html("image.jpg" if image_url else None, "music.mp3" if music_url else None, parsed["bg_color"])

print("\n✅ Sayt uğurla yaradıldı! Qovluğa bax: xeyalin_sayti")