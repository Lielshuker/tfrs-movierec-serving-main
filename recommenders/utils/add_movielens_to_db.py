import sys
from pathlib import Path
from imdb import IMDb
# from app import db
# # from app.models import Movie, User, UserMovieRating
# from app.models import Movie
# import re
#
# # https://m.media-amazon.com/images/M/MV5BNTFkZjdjN2QtOGE5MS00ZTgzLTgxZjAtYzkyZWQ5MjEzYmZjXkEyXkFqcGdeQXVyMTM0NTUzNDIy._V1_.jpg
# # https://m.media-amazon.com/images/M/MV5BNDc3Y2YwMjUtYzlkMi00MTljLTg1ZGMtYzUwODljZTI1OTZjXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX101_CR0,0,101,150_.jpg
# movies = Movie.query.filter().all()
# for movie in movies:
#     if '@._V1_' in movie.image_url:
#         a = re.sub('._V1_.*?.jpg', '', movie.image_url, flags=re.DOTALL) + '.__V1__.jpg'
#         movie.image_url = a
#         db.session.add(movie)
#         db.session.commit()
#         # DATA_DIRNAME = Path(__file__).resolve().parents[2] / 'data'
# # print(DATA_DIRNAME)
#
# ia = IMDb()
# links = DATA_DIRNAME /'links.csv'
# ratings = DATA_DIRNAME/ 'ratings.csv'
# from datetime import date
#
# today = date.today()
# with open(links,'r') as f:
#     line = f.readline().strip()
#     n = 0
#     while True:
#         n+=1
#         line = f.readline().strip()
#         if n < 9731:
#             continue
#         if not line:
#             break
#
#         movielens, imdb, tmdb = line.split(',')
#         movie = ia.get_movie(imdb)
#         try:
#             title = movie.get('title','')
#             cover = movie.get('cover url','')
#             year = movie.get('year',0)
#             plot = movie.get('plot outline')
#             # description = movie.get('description')
#
#             length = movie.get('runtimes', [])
#             if length:
#                 length = length[0]
#
#             date_added = today.strftime("%Y-%m-%d")
#             if title == '' or cover == '' or year == 0 or plot is None or length is []:
#                 print("error in", n, ":title == '' or cover == '' or year == 0 or plot is None or length is []")
#
#                 continue
#
#             m = Movie(name=title, length=length, release_year=year, date_added=date_added,
#                     description=plot, movie_url=None, image_url=cover, movielens_id=int(movielens))
#             # m = Movie(imdbid = int(imdb),movielensid = int(movielens),\
#             #     title = title, cover = cover, plot = plot, year = year)
#             db.session.add(m)
#             db.session.commit()
#
#             if n % 100 == 0:
#                 print(n, m)
#         except:
#             print("error in n", n)
#
# db.session.commit()

# with open(ratings, 'r') as f:
#     line = f.readline().strip()
#     n = 0
#     while True:
#         n+=1
#         line = f.readline().strip()
#         if not line:
#             break
#         username, movielensid, rating, _ = line.split(',')
#         user = User.query.filter_by(username = username).first()
#         movie = Movie.query.filter_by(movielensid = int(movielensid)).first()
#         if movie is None:
#             continue
#         if user is None:
#             user = User(username=username)
#             db.session.add(user)
#         r = UserMovieRating(movieid = movie.movieid, userid = user.userid, rating = float(rating))
#         # r.Movie = movie
#         user.movies.append(r)
#         movie.users.append(r)
#         db.session.add(r)
#
#
#         if n % 100 == 0:
#             print(n, r)
#
# db.session.commit()
#
#
#
#
#
#
#
#
#
