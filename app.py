from flask import Flask, request, render_template
from backend import user_input

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    result = None
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form["text"]
        if not input_text:
            prediction="Model says: Neutral"
        result = user_input(input_text)
        if 0.4 < result < 0.6:
         prediction="Model says: Neutral"
        elif result >= 0.6:
         prediction="Model says: Positive"
        else:
         prediction="Model says: Negative"
    return render_template("home.html", prediction=prediction,input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
