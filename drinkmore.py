import streamlit as st
import pandas as pd

current_temp = 28
prev_temp = 0
next_reminder = ["09:00:00", "09:40:00", "10.20:00", "11:00:00", "11:40:00"]

home, schedule = st.tabs(["Home", "Schedule"])

with home:
    st.title("HOME")
    st.metric("Current Temperature", current_temp, prev_temp, "inverse")
    recheck = st.button("Re-check Temperature")

    if recheck:
        # updateTemp()
        st.write("Temperature updated")

    pop = st.popover("Next reminder")
    pop.write(next_reminder[0])
    pop.write(next_reminder[1])
    pop.write("See complete schedule in \"Schedule\" tab")

with schedule:
    st.title("SCHEDULE")
    
    new_reminder = st.time_input("Add reminder")
    confirm = st.button("Confirm")

    if confirm:
        next_reminder.append(new_reminder)
        st.write("New reminder successfully added")

    data = {"Time": next_reminder}
    df = pd.DataFrame(data)

    st.dataframe(df)