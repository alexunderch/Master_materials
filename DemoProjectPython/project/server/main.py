from flask import Flask, render_template
import redis    
from typing import Literal
import os, sys
app = Flask(__name__)


db = redis.from_url("redis://redis:6969/0")
import sys, os, json


def create_db() -> None:
    def get_exp_path(prefix: str, 
                     use_sample_pool: bool, 
                     mutate_pool: bool, 
                     loss_type: Literal["ce", "l2"], 
                     add_noise: bool):
        """A helpful funtion which configures all paths to all models"""
        path = prefix + '/'
        path += 'use_sample_pool_%r mutate_pool_%r '%(use_sample_pool, mutate_pool)
        path += 'loss_type_%s '%(loss_type)
        path += 'add_noise_%r'%(add_noise)
        path += '/0100000.json'
        return path

    prefix = '/model_data'
    for use_sample_pool in [False, True]:
        for mutate_pool in [False, True]:
            for loss_type in ["l2", "ce"]:
                for add_noise in [False, True]:
                    if use_sample_pool == False and mutate_pool ==  True:
                        #hardcode
                        continue
                    key = (use_sample_pool, mutate_pool, loss_type, add_noise)
                    db.set(str(key), get_exp_path(prefix, use_sample_pool, mutate_pool, loss_type, add_noise))

def make_model(key: str) -> None:
    name =  db.get(key).decode("utf-8")
    with open(name, "r") as f:
        jaux = json.load(f)
    with open(
        os.path.join("/app", "server", "static", "model.json"), "w+"
            ) as f:
        json.dump(jaux, f)

@app.route("/")
def index():
    return render_template('main_page.html')


@app.route("/mutating")
def mutating():
    create_db()
    make_model("(True, True, 'ce', True)")
    return render_template('index.html')


@app.route("/persistent")
def persistent():
    create_db()
    make_model("(True, False, 'l2', True)")
    return render_template('index.html')

@app.route("/naive")
def naive():
    create_db()
    make_model("(False, False, 'l2', True)")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5555)