from app import db


# class UserMovieRating(db.Model):
#     userid = db.Column(db.Integer, db.ForeignKey('user.userid'), primary_key=True)
#     movieid = db.Column(db.Integer, db.ForeignKey('movie.movieid'), primary_key=True)
#     rating = db.Column(db.Float)
#     user = db.relationship('User', back_populates='movies')
#     movie = db.relationship('Movie', back_populates='users')
#
#     def __repr__(self):
#         return '<User {}, Movie {}, Rating {}>'.format(self.user.username, self.movie.title, self.rating)
#
class User(db.Model):
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    # movies = db.relationship('UserMovieRating', back_populates='user')

    def __repr__(self):
        return '<User {}>'.format(self.username)

    # def is_watched(self, movie):
    #     for r in self.movies:
    #         if r.movieid == movie.movieid:
    #             return r
    #     return None


# class Movie(db.Model):
#     movieid = db.Column(db.Integer, primary_key=True)
#     movielensid = db.Column(db.Integer, index=True, unique=True)
#     imdbid = db.Column(db.Integer, index=True, unique=True)
#     title = db.Column(db.String(64), index=True)
#     cover = db.Column(db.String)
#     plot = db.Column(db.String)
#     year = db.Column(db.Integer)
#     users = db.relationship('UserMovieRating', back_populates='movie')
#
#     def __repr__(self):
#         return '<Movie {}>'.format(self.title)
class Movie(db.Model):
    __tablename__ = 'movies'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(512))
    length = db.Column(db.Integer)
    #genre_id = db.Column(db.Integer)
    genre_id = []
    release_year = db.Column(db.Date)
    date_added = db.Column(db.Date)
    description = db.Column(db.Text)
    movie_url = db.Column(db.String(512))
    image_url = db.Column(db.String(512))
    movielens_id = db.Column(db.Integer)

    def __int__(self, name, length, genre_id, release_year, date_added,
                description, movie_url, image_url, movielens_id):
    #def __int__(self, name, length, release_year, date_added,
    #            description, movie_url, image_url):
        self.name = name
        self.length = length
        self.genre_id = genre_id
        self.release_year = release_year
        self.date_added = date_added
        self.description = description
        self.movie_url = movie_url
        self.image_url = image_url
        self.movielens_id = movielens_id

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class WatchedMovie(db.Model):
    __tablename__ = 'watched_movies'
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer)
    user_id = db.Column(db.Integer)
    watch_num = db.Column(db.Integer)
    date_modify = db.Column(db.DATE)
    rate = db.Column(db.Integer)

    def __init__(self, movie_id, user_id, watch_num, date_modify, rate):
        self.movie_id = movie_id
        self.user_id = user_id
        self.watch_num = watch_num
        self.date_modify = date_modify
        self.rate = rate

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
