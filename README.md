# Tech-Recommodation

🧠 Tech Product Recommender (Tkinter App)

An intelligent desktop application that recommends similar tech products based on product names, categories, and descriptions — built with Python, Tkinter, and scikit-learn.

The app automatically loads all .csv or .xlsx files from your Dataset folder, merges them into a single searchable list, and allows you to:

Browse by category (each file becomes a category)

Search for a specific product

Get Top-K similar product recommendations

View product details (title, category, price, rating)

Display the product image (from URL in the image column)

Open the product link in your browser

🧩 Features

✅ Merge multiple datasets automatically
✅ Intelligent text similarity (TF-IDF + cosine similarity)
✅ Visual interface using Tkinter
✅ Image preview from the dataset
✅ Direct product link button
✅ Category filtering and live search

🗂️ Folder Structure
Tech Product Recommendation/
│
├── recommender_app.py          # Main Tkinter app
├── Dataset/                    # Folder containing your CSV/XLSX files
│   ├── All Electronics.csv
│   ├── Cameras.xlsx
│   └── ...
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore                  # Ignore virtual envs and datasets

⚙️ Installation
1. Clone the repository
git clone https://github.com/<your-username>/tech-product-recommendation.git
cd tech-product-recommendation

2. Create a virtual environment (recommended)
python -m venv venv

3. Activate it

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

▶️ Usage

Make sure your dataset files (.csv or .xlsx) are inside:

Tech Product Recommendation/Dataset/


Each file should contain these columns (at least):

name, main_category, sub_category, image, link, ratings, discount_price, actual_price


The application automatically maps them internally.

Run the application:

python recommender_app.py


When it opens:

Select a category (each dataset file)

Search for a product

Click Recommend

See similar items and product images

Click Open Link to view the product online

🧠 How It Works

The app merges all datasets into one table.

Combines title, category, and description into a single text field.

Uses TF-IDF vectorization and cosine similarity to compute relationships between products.

Returns the Top-K most similar items to any selected product.

🧰 Technologies Used

Python 3.11+

Tkinter (GUI)

Pandas (data handling)

scikit-learn (TF-IDF + cosine similarity)

Pillow (PIL) (image handling)

Requests (fetching online images)
