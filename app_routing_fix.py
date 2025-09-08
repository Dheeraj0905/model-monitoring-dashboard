# Fixed routing section for app.py
# Replace lines 147-163 with this corrected version

    # Route to appropriate page based on current_page state
    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "📁 Model Upload":
        show_model_upload_page()
    elif st.session_state.current_page == "📊 Dataset Upload":
        show_dataset_upload_page()
    elif st.session_state.current_page == "📋 Schema Definition":
        show_schema_definition_page()
    elif st.session_state.current_page == "📊 Raw Data Testing":
        show_raw_data_testing_page()
    elif st.session_state.current_page == "🎲 Data Generation":
        show_data_generation_page()
    elif st.session_state.current_page == "⚡ Performance Testing":
        show_performance_testing_page()
    elif st.session_state.current_page == "📊 Results & Analytics":
        show_results_page()
    elif st.session_state.current_page == "📈 Explainability":
        show_explainability_page()
