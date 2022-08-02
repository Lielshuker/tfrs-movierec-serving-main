from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor


db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    # db = SQLAlchemy(app)
    # db.init_app(app)
    migrate = Migrate(app, db)
    return app


executors = {
    'default': ThreadPoolExecutor(16),
    'processpool': ProcessPoolExecutor(4)
}
from recommenders.models.scheduler_predict import job_predict

sched = BackgroundScheduler(executors=executors)
sched.add_job(job_predict, 'interval', hours=24)

#
# from app import  models