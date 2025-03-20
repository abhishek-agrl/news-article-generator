from flask import Flask, request, render_template, Response
from article_generator import ArticleGenerator
import config

app = Flask(__name__)
article_generator = ArticleGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    search_query = request.form['search_query']
    response_text = article_generator.generate(search_query=search_query.strip())
    if not response_text:
        return Response(status=500, response="Failed to generate, likely due to missing context")
    return response_text

if __name__ == '__main__':
    app.run(debug=True)