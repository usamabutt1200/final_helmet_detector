                "▶ Process Video",
                type="primary",
                use_container_width=True,
            )

            if process_video:
                temp_video_path = None
                capture = None

                try:
                    video_bytes = uploaded_video.getvalue()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded_video.name)[1] or ".mp4",
                    ) as temp_video:
                        temp_video.write(video_bytes)
                        temp_video_path = temp_video.name

                    capture = cv2.VideoCapture(temp_video_path)

                    if not capture.isOpened():
                        st.error("The uploaded video could not be opened.")
                    else:
                        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        total_frames = max(total_frames, 1)

                        progress = st.progress(0)
                        status = st.empty()
                        frame_display = st.empty()

                        frame_number = 0

                        while True:
                            success, frame = capture.read()

                            if not success:
                                break

                            predictions = neural_engine.predict(
                                source=frame,
                                conf=video_confidence,
                                verbose=False,
                            )

                            annotated_frame = predictions[0].plot()
                            annotated_rgb = cv2.cvtColor(
                                annotated_frame,
                                cv2.COLOR_BGR2RGB,
                            )

                            frame_display.image(
                                annotated_rgb,
                                channels="RGB",
                                use_container_width=True,
                            )

                            frame_number += 1
                            progress.progress(
                                min(frame_number / total_frames, 1.0)
                            )
                            status.write(
                                f"Processed frame {frame_number} of {total_frames}"
                            )

                        st.success(
                            f"Video processing complete. Total frames processed: {frame_number}"
                        )

                except Exception as error:
                    st.error(f"Video processing failed: {error}")

                finally:
                    if capture is not None:
                        capture.release()

                    if temp_video_path and os.path.exists(temp_video_path):
                        os.remove(temp_video_path)


# ============================================================
# WORKSPACE 4: BROWSER LIVE CAMERA
# ============================================================
elif workspace == "Real-Time Live Feed Stream":
    st.title("📡 Real-Time Live Camera Detection")
    st.caption(
        "Use your browser camera to run helmet detection on a live video stream."
    )

    live_confidence = st.slider(
        "Live detection confidence",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
        key="live_confidence",
    )

    class HelmetVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.confidence = live_confidence

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")

            predictions = neural_engine.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
            )

            annotated = predictions[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="vanguard-helmet-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HelmetVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.caption(
        "If the camera does not start, allow camera permission in your browser. "
        "Live camera access requires HTTPS when hosted online."
