from flask import Flask, request, render_template
from article_generator import ArticleGenerator

app = Flask(__name__)
article_generator = ArticleGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    search_query = request.form['search_query']
    response_text = article_generator.generate(search_query=search_query.strip())
    return response_text

if __name__ == '__main__':
    app.run(debug=True)