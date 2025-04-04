import streamlit as st
import sys
import os

# Добавляем пути к подпроектам
sys.path.insert(0, os.path.abspath("face"))
sys.path.insert(0, os.path.abspath("forest"))
sys.path.insert(0, os.path.abspath("ship"))

# Импортируем модули приложений
from pages.face_app import face_app
from pages.forest_app import forest_app
from pages.ship_app import ship_app

def main():
    st.sidebar.title("Навигация")
    app_mode = st.sidebar.selectbox("Выберите приложение", 
                                  ["Face Detection", 
                                   "Forest Analysis", 
                                   "Ship Recognition"])

    if app_mode == "Face Detection":
        face_app()
    elif app_mode == "Forest Analysis":
        forest_app()
    elif app_mode == "Ship Recognition":
        ship_app()

if __name__ == "__main__":
    main()