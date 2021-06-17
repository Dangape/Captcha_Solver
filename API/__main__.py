from API import create_app
import os

app = create_app()

def start_server():
    port  = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    return True

if __name__ == '__main__':
    start_server()