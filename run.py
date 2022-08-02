from app import create_app, sched

app = create_app()
def main():
    sched.start()
    app.run(host="0.0.0.0", port=8000, debug=False)

if __name__ == "run.py":
    main()