import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import random
import time
from datetime import datetime
import threading
import pytz

def build_model():
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    model = Sequential()
    model.add(vgg19_base)
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam')
    return model

def preprocess_frame(image):
    img = cv2.resize(image, (160, 160))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def app():
    if "current_temp" not in st.session_state:
        st.session_state["current_temp"] = 25.0
    if "prev_temp" not in st.session_state:
        st.session_state["prev_temp"] = 9999
    if "update_temp" not in st.session_state:
        st.session_state["update_temp"] = False
    if "reminder_green" not in st.session_state:
        st.session_state["reminder_green"] = ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "19:00", "21:00"]
    if "reminder_yellow" not in st.session_state:
        st.session_state["reminder_yellow"] = ["09:00", "09:40", "10.20", "11:00", "11:40", "12:20", "13:00", "13:40", "14:20", "15:00", "15:40", "16:20", "17:00", "18:00", "19:00", "20:00", "21:00"]
    if "reminder_red" not in st.session_state:
        st.session_state["reminder_red"] = ["09:00", "09:15", "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45", "16:00", "16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45", "18:00", "19:00", "18:15", "18:30", "18:45", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45"]
    if "state" not in st.session_state:
        st.session_state["state"] = "reminder_green"
    if "temp_state" not in st.session_state:
        st.session_state["temp_state"] = "temp_state"
    if "update_reminder" not in st.session_state:
        st.session_state["update_reminder"] = False
    if "status_reminder" not in st.session_state:
        st.session_state["status_reminder"] = []
    if "update_status" not in st.session_state:
        st.session_state["update_status"] = False
    if "model" not in st.session_state:
        model = build_model()
        model.build(input_shape=(None, 160, 160, 3))
        model.load_weights("vgg19_cpu.weights.h5")
        st.session_state["model"] = model
    if "drinking" not in st.session_state:
        st.session_state["drinking"] = False

    st.markdown(
        """
        <style>
        .st-emotion-cache-gi0tri.e1nzilvr1 {display: none}
        .st-emotion-cache-1xarl3l.e1i5pmia1{
            width: 170px;
            height: 170px;
            border-radius: 50%;
            border: 5px solid black;
            display: flex;
            background-color: rgb(237, 237, 237);
            align-items: center;
            align-content: center;
            justify-content: center;
            font-size: 40px;
            font-weight: bold;
            margin: auto;
        }
        .st-emotion-cache-1xarl3l.e1i5pmia1.st-emotion-cache-1wivap2.e1i5pmia3{
            display: flex;
            justify-content: center;
        }
        .st-emotion-cache-17c4ue.e1i5pmia2{
            display: flex;
            justify-content: center;
            margin-bottom: 5px;
        }
        .st-emotion-cache-1whx7iy.e1nzilvr4 p{
            font-size: 25px;
        }
        .st-emotion-cache-wnm74r.e1i5pmia0{
            margin-top: 5px;
            display: flex;
            justify-content: center;
            align-content: center;
            font-size: 20px;
        }
        .st-emotion-cache-1wivap2.e1i5pmia3{
            padding: 0;
            margin: 0;
            width: auto;
        }
        .st-emotion-cache-ocqkz7.e1f1d6gn5{
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,
        options=["Home", "Schedule", "Drinking Detection"],
        icons=["house", "alarm", "robot"],
        default_index=0,
        orientation="horizontal"
    )

    # Home Tab
    if selected == "Home":
        st.session_state["update_reminder"] = False

        st.title("HOME")
        st.write("---")
        st.metric(label="Current Temperature",
                value=str(round(st.session_state["current_temp"], 1)) + "°C", 
                delta= str(round(st.session_state["current_temp"] - st.session_state["prev_temp"], 1)) + "°C" if st.session_state["prev_temp"]!=9999 else None,
                delta_color="inverse",
                )
        if(st.session_state["current_temp"] <= 25):
            st.success("Ambient temperature is normal. Don't forget to drink as scheduled! :smile:")
            st.session_state["state"] = "reminder_green"

        if(st.session_state["current_temp"] > 25 and st.session_state["current_temp"] < 29):
            st.warning("Ambient temperature is quite high. Don't forget to stay hydrated! :droplet:")
            st.session_state["state"] = "reminder_yellow"

        if(st.session_state["current_temp"] >= 29):
            st.error("Ambient temperature is above normal limit. Please drink more often! :fire:")
            st.session_state["state"] = "reminder_red"

        col1, col2 = st.columns([2, 5])
        recheck = col1.button("Re-check Temperature")

        if recheck:
            st.session_state["update_temp"] = True

            progress_bar = st.progress(0)
            for i in range(1, 101):
                progress_bar.progress(i)
                time.sleep(0.02)

            st.session_state["prev_temp"] = st.session_state["current_temp"]
            st.session_state["current_temp"] = random.uniform(15.0, 37.0)
            st.rerun()

        if st.session_state["update_temp"] == True:
            col2.success("Temperature updated")

        st.write("")
        st.write("---")

        jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
        next_reminder = list(filter(lambda x: x > jkt.strftime("%H:%M"), st.session_state[st.session_state["state"]]))
        # next_reminder = []

        pop = st.popover("Next reminder")
        if len(next_reminder) > 1:
            pop.write("- " + next_reminder[0])
            pop.write("- " + next_reminder[1])
            pop.write("See complete schedule in \"Schedule\" tab")
        elif len(next_reminder) > 0:
            pop.write("- " + next_reminder[0])
            pop.write("-")
            pop.write("See complete schedule in \"Schedule\" tab")
        else:
            pop.write("No more schedule for today! :grin::thumbsup:")
            pop.write("See you tomorrow! :wave:")

        st.write("")
        st.write("")
        st.write("")

    # Schedule Tab
    if selected == "Schedule":
        st.session_state["update_temp"] = False

        st.title("SCHEDULE")
        st.write("---")

        jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
        next_reminder = list(filter(lambda x: x > jkt.strftime("%H:%M"), st.session_state[st.session_state["state"]]))
        if st.session_state["temp_state"] != st.session_state["state"] or st.session_state["update_status"] == True:
            status_reminder = []
            # for i in st.session_state[st.session_state["state"]]:
            #     status_reminder.append("Holycow")
            for x in st.session_state[st.session_state["state"]]:
                if x not in next_reminder:
                    status_reminder.append("❌ Missed" if random.random() < 0.2 else "✔️ Done")
                else:
                    status_reminder.append("⏱️ Pending")
            st.session_state["status_reminder"] = status_reminder
            st.session_state["temp_state"] = st.session_state["state"]

        data = {
            "Time": st.session_state[st.session_state["state"]],
            "Status": st.session_state["status_reminder"],
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.write("---")
        new_reminder = st.time_input("Add reminder")
        sch_col1, sch_col2 = st.columns([1, 5])
        confirm = sch_col1.button("Confirm")

        if confirm:
            st.session_state["update_reminder"] = True
            st.session_state["update_status"] = True

            str_new_reminder = str(new_reminder)
            list_state = ["reminder_green", "reminder_yellow", "reminder_red"]
            for x in list_state:
                temp_list = list(st.session_state[x])
                temp_list.append(str_new_reminder[:-3])
                temp_list.sort()
                st.session_state[x] = temp_list
            
            st.rerun()
        st.session_state["update_status"] = False    
        
        if st.session_state["update_reminder"] == True:
            sch_col2.success("New reminder successfully added")

    # Detection Tab
    if selected == "Drinking Detection":
        model = st.session_state["model"]
        st.session_state["update_temp"] = False
        st.session_state["update_reminder"] = False

        st.title("Drinking Detection")
        option = st.selectbox(
            label="Drinking Detection",
            options=["Real-Time Detection Using Webcam", "Upload Image File"],
            index=None,
            placeholder="Select a method...",
            label_visibility="collapsed",
        )

        if option == "Real-Time Detection Using Webcam":
            lock = threading.Lock()
            img_container = {"img": None}

            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                with lock:
                    img_container["img"] = img
                return frame

            ctx = webrtc_streamer(
                key="example",
                video_frame_callback=video_frame_callback,
                rtc_configuration={  # Remove this if webcam does not work properly
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )
            fig_place = st.empty()

            while ctx.state.playing:
                with lock:
                    img = img_container["img"]
                if img is None:
                    continue
                x = preprocess_frame(img)
                prediction = model.predict(x)

                if prediction[0][0] > 0.8:
                    st.session_state["drinking"] = True
                
                if st.session_state["drinking"] == True:
                    fig_place.success("Completed!")
            
            if st.session_state["drinking"] == True:
                st.success("Congratz! You have stayed hydrated! :thumbsup:")
                st.write("Waiting for the next reminder! :stopwatch:")
                reset = st.button("Reset Status", help="Reset the current \"completed\" status to \"pending\"")
                if reset:
                    st.session_state["drinking"] = False
            else:
                st.warning("Complete current reminder by starting the webcam up there! :point_up_2:")
        
        elif option == "Upload Image File":
            file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
            
            if file is not None:
                file_image = Image.open(file)
                st.image(image=file_image, caption="Uploaded Image", use_column_width=True)

                bytes_data = file.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                x = preprocess_frame(cv2_img)
                prediction = model.predict(x)

                if np.argmax(prediction) == 0 or prediction[0][0] > 0.2:
                    st.success("Drinking")
                else:
                    st.warning("Not Drinking")


if __name__ == "__main__":
    app()