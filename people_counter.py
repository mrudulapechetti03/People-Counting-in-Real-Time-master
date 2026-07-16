from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import argparse
import datetime
import schedule
import logging
import imutils
import time
import json
import csv
import cv2

# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

# load config
with open("utils/config.json", "r") as file:
    config = json.load(file)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4)
    return vars(ap.parse_args())


def send_mail():
    if not (
        config["Email_Send"]
        and config["Email_Receive"]
        and config["Email_Password"]
    ):
        logger.warning("Email configuration missing.")
        return

    try:
        Mailer().send(config["Email_Receive"])
        logger.info("Alert email sent.")
    except Exception as e:
        logger.error(f"Email could not be sent: {e}")


def log_data(move_in, in_time, move_out, out_time):
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue='')
    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
            wr.writerows(export_data)


def people_counter():
    print(">>> RUNNING NODLIB VERSION")

    args = parse_arguments()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Load MobileNetSSD
    logger.info("Loading Caffe model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    logger.info("Model loaded successfully.")

    # Initialize stream
    if args.get("input", None) is None:
        logger.info("Starting the live stream..")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
        is_file_input = False
    else:
        logger.info(f"Starting the video file: {args['input']}")
        vs = cv2.VideoCapture(args["input"])
        if not vs.isOpened():
            logger.error(f"Cannot open video file: {args['input']}")
            return
        is_file_input = True

    # Only use threaded class for live stream
    if config.get("Thread") and not is_file_input:
        logger.info("Using threaded camera input.")
        vs = thread.ThreadingClass(config["url"])

    writer = None
    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    people_inside = 0
    alert_sent = False
    move_out = []
    move_in = []
    out_time = []
    in_time = []

    fps = FPS().start()

    logger.info("Entering main loop (NO dlib, detection + centroid tracking only)...")

    while True:
        # ----------------- READ FRAME -----------------
        if is_file_input:
            grabbed, frame = vs.read()
            if not grabbed or frame is None:
                logger.info("No more frames from video. Exiting loop.")
                break
        else:
            frame = vs.read()
            if frame is None:
                logger.error("Frame is None from live stream. Exiting loop.")
                break

        frame = imutils.resize(frame, width=500)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            logger.info(f"Frame size: W={W}, H={H}")

        # NEW LINE POSITION (30% height)
        line_y = int(H * 0.3)

        # Initialize writer if requested
        if args.get("output") and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
            logger.info(f"Output video writer created: {args['output']}")

        status = "Detecting"
        rects = []

        # --------------- PERSON DETECTION (EVERY FRAME) ---------------
        blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843,
                                     size=(300, 300), mean=127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx >= len(CLASSES) or CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # Clamp box to frame bounds
                startX = max(0, min(W - 1, startX))
                startY = max(0, min(H - 1, startY))
                endX = max(0, min(W - 1, endX))
                endY = max(0, min(H - 1, endY))

                # Skip invalid boxes
                if endX <= startX or endY <= startY:
                    continue

                rects.append((startX, startY, endX, endY))

                # Draw detection box
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

        # --------------- DRAW LINE ----------------
        cv2.line(frame, (0, line_y), (W, line_y), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border-", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # --------------- UPDATE TRACKER (CentroidTracker only) ---------------
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    # EXIT (moving upward)
                    if direction < 0 and centroid[1] < line_y:
                        totalUp += 1
                        move_out.append(totalUp)
                        out_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

                        people_inside = len(move_in) - len(move_out)
                        to.counted = True

                    # ENTER (moving downward)
                    elif direction > 0 and centroid[1] > line_y:
                        totalDown += 1
                        move_in.append(totalDown)
                        in_time.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

                        people_inside = len(move_in) - len(move_out)
                        to.counted = True
            trackableObjects[objectID] = to

            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4,
                       (255, 255, 255), -1)

        # Info text
        info_status = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status)
        ]

        info_total = [("Total people inside", people_inside)]

        for (i, (k, v)) in enumerate(info_status):
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info_total):
            text = f"{k}: {v}"
            cv2.putText(frame, text, (250, H - ((i * 20) + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# ================= THRESHOLD CHECKER =================

        if people_inside >= config["Threshold"]:

            cv2.putText(frame,
                    "ALERT! THRESHOLD EXCEEDED",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

            if config["ALERT"] and not alert_sent:
                send_mail()
                logger.info("Alert email sent.")
                alert_sent = True

        else:
            alert_sent = False

# =====================================================


        if config.get("Log"):
            log_data(move_in, in_time, move_out, out_time)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Q pressed, exiting.")
            break

        totalFrames += 1
        fps.update()

    # stop the FPS counter
    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    # release resources
    try:
        if is_file_input and hasattr(vs, "release"):
            vs.release()
        elif hasattr(vs, "stop"):
            vs.stop()
    except:
        pass

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    logger.info("People counter finished.")


if __name__ == "__main__":
    # ignore Scheduler for now; run directly
    people_counter()
