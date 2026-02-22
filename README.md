# საგადასახადო RAG ჩატბოტი

საქართველოს შემოსავლების სამსახურის ინფოჰაბის მონაცემთა ბაზაზე დაფუძნებული AI ასისტენტი. მუშაობს chromedb ვექტორულ მონაცემთა ბაზაზე + Gemini-flash-2.5 
ვექტორული ბაზის ასაწყობად, ვიყენებთ BGE M3 Embedding-ს.

## ინსტალაცია

```powershell
# ვირტუალური გარემო
python -m venv venv
.\venv\Scripts\activate

# დამოკიდებულებები
pip install -r requirements.txt
```

## კონფიგურაცია

1. შექმენით `.env` ფაილი:
```powershell
copy .env.example .env
```

2. ჩასვით Gemini API გასაღები `.env` ფაილში:
```
GEMINI_API_KEY=api_გასაღები
```

## გაშვება

### 1. მონაცემთა ბაზის შექმნა (პირველი ჯერზე)

```powershell
python crawler.py
```

შესაძლოა დაჭირდეს საათზე მეტი.

### 2. ჩატბოტის გაშვება

```powershell
python main.py
```

ჩატბოტთან ინტერაქცია არის ქართულ ენაზე.

გამოსასვლელად: `exit` ან `გამოსვლა`

## სტრუქტურა

```
├── main.py              # მთავარი პროგრამა
├── config.py            # კონფიგურაცია
├── vector_store.py      # ვექტორული ძიება
├── agent.py             # AI აგენტი
├── crawler.py           # მონაცემთა შემოტანა
└── .env                 # API გასაღები (შექმენი თავად)
```

## მზა ვექტორული მონაცემთა ბაზის ჩამოტვირთვა (ასევე მოდელის)
https://huggingface.co/datasets/terminalvelocity/chroma_infohub_bgem3_full

## პრობლემების შემთხვევაში:

- **"GEMINI_API_KEY not set"** → შექმენი `.env` ფაილი API გასაღებით
- **"Vector database not found"** → გაუშვი `crawler.py` ჯერ
- **"Corpus file not found"** → დაელოდე `crawler.py`-ის დასრულებას
