import argparse
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from models import YOLOModel, EnsembleModel
from image_processing import ImageProcessor
from training import Trainer
from HPC_integration import parallel_model_training, data_parallelism, hyperparameter_tuning_hpc
from deploy import app as flask_app
from video_processor import VideoProcessor
from ship_tracker import ShipTracker
from round_counter import RoundCounter
from result_generator import ResultGenerator


def handle_error(e, critical=False):
    # process stops if critical, otherwise it shows warning and continue
    if critical:
        print(f"Critical error: {str(e)}")
        raise e
    else:
        print(f"Non-critical error: {str(e)}")


def train_model(args):
    model = YOLOModel(model_name='yolov8n', num_classes=6) # buoys, numbers, model with/without number, trace of model, special buoy
    
    if args.use_ddp:
        model = parallel_model_training(model, args.world_size) # distributes data parallelization, if possible
    elif args.use_data_parallel:
        model = data_parallelism(model)
    
    # training.py
    trainer = Trainer(model, args.train_dir, args.num_epochs, args.output_dir, world_size=args.world_size, patience=args.patience)
    best_model = trainer.train()
    

    if args.optimize:
        trainer.optimize(n_trials=args.n_trials, num_nodes=args.num_nodes, node_rank=args.node_rank) # proveides optimized hyperparameter
        print(f"the best hyperparameters for training: {trainer} \n !")

    torch.save(best_model.state_dict(), f"{args.output_dir}/best_model.pth") # save the best one in the same dir
    print(f"Best model saved to {args.output_dir}/best_model.pth")


def process_images(args):

    model = YOLOModel(model_name='yolov8n', num_classes=6)
    processor = ImageProcessor(model, input_dir=args.input_dir, output_dir=args.output_dir) # image_processing.py
    processor.process_and_save_images()


def analyze_image(args):
    model = YOLOModel(model_name='yolov8n', num_classes=6)
    processor = ImageProcessor(model) # image_processing.py
    processor.analyze_image(args.image_path)


def deploy_model(args):
    model = YOLOModel(model_name='yolov8n', num_classes=6)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    flask_app.config['MODEL'] = model # deploying by Flask API (trained mode)
    flask_app.run(host=args.host, port=args.port)


def process_race(args):
 
    # split the inputted video by photos and processing them (edge computing)
    
    model = YOLOModel(model_name='yolov8n', num_classes=6)
    processor = ImageProcessor(model) # image_processing.py
    tracker = ShipTracker() # ship_tracker
    counter = RoundCounter(num_buoys=args.num_buoys) # round_counter.py
    video_proc = VideoProcessor(args.camera_inputs) # video_processor.py
    result_gen = ResultGenerator()

    def process_frame(frame, camera_id):
        """
        Process a single frame from a camera.
        
        This function detects ships, tracks them, updates round counts,
        and generates annotated frames.
        
        Args:
        frame: The input frame
        camera_id: ID of the camera that captured this frame
        """
        try:
            detections = processor.process_frame(frame)
            tracked_objects = tracker.update(detections)
            counter.update(tracked_objects, camera_id)
            
            # Detect colors in the first 10 seconds of the race
            if video_proc.elapsed_time < 10:
                colors = processor.detect_colors(tracked_objects)
                tracker.update_colors(colors)
            
            annotated_frame = processor.annotate_frame(frame, tracked_objects)
            video_proc.write_frame(annotated_frame, camera_id)
            video_proc.display_frame(annotated_frame)

            # Generate and display live leaderboard every minute
            if video_proc.elapsed_time % 60 == 0:
                leaderboard = result_gen.generate_live_leaderboard(counter.get_results())
                print("Current Leaderboard:")
                for ship_id, rounds in leaderboard:
                    print(f"Ship {ship_id}: {rounds} rounds")

        except Exception as e:
            print(f"Error processing frame from camera {camera_id}: {str(e)}")

    try: # parallel processing of camera's frame
        with ThreadPoolExecutor(max_workers=3) as executor: 
            for frame, camera_id in video_proc.get_frames():
                executor.submit(process_frame, frame, camera_id)

        results = counter.get_results() # final result (number of completed rounds for each of racers) 
        result_gen.generate(results, video_proc.get_output_videos()) # final result (outputted video with emphasized classes)
    except Exception as e:
        print(f"An error occurred during race processing: {str(e)}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="YOLOv8n Object Detection Project")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model") # training
    train_parser.add_argument("--train_dir", required=True, help="Directory with training images")
    train_parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--output_dir", default="./output", help="Output directory")
    train_parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    train_parser.add_argument("--n_trials", type=int, default=10, help="Number of optimization trials")
    train_parser.add_argument("--use_ddp", action="store_true", help="Use Distributed Data Parallel")
    train_parser.add_argument("--use_data_parallel", action="store_true", help="Use Data Parallelism")
    train_parser.add_argument("--world_size", type=int, default=1, help="Number of processes for DDP")
    train_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for hyperparameter tuning")
    train_parser.add_argument("--node_rank", type=int, default=0, help="Rank of the current node")
    train_parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")

    process_parser = subparsers.add_parser("process", help="Process images") # process images
    process_parser.add_argument("--input_dir", required=True, help="Input directory with images")
    process_parser.add_argument("--output_dir", required=True, help="Output directory for processed images")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single image") # analyze images
    analyze_parser.add_argument("--image_path", required=True, help="Path to the image to analyze")

    deploy_parser = subparsers.add_parser("deploy", help="Deploy model as Flask API") # deploy parser in local host&port
    deploy_parser.add_argument("--model_path", required=True, help="Path to the saved model")
    deploy_parser.add_argument("--host", default="0.0.0.0", help="Host to run the Flask app")
    deploy_parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app")

    race_parser = subparsers.add_parser("race", help="Process a ship race") # race processing parser
    race_parser.add_argument("--camera_inputs", nargs=3, required=True, help="Paths to the three camera inputs")
    race_parser.add_argument("--num_buoys", type=int, required=True, help="Number of buoys in the race course")

    args = parser.parse_args()

    try:
        if args.command == "train":
            train_model(args)
        elif args.command == "process":
            process_images(args)
        elif args.command == "analyze":
            analyze_image(args)
        elif args.command == "deploy":
            deploy_model(args)
        elif args.command == "race":
            process_race(args)
        else:
            parser.print_help()
    except Exception as e:
        handle_error(e, critical=True)
