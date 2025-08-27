<!-- PTZ Camera Tracking System Instructions -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	<!-- Python PTZ camera tracking system with ONVIF control, real-time body tracking using OpenCV, OSC/API control interface, multiple person detection and tracking with smoothing algorithms, preset management, and Bitfocus Companion integration for Windows -->

- [x] Scaffold the Project
	<!-- Created project structure with main modules: tracker.py, camera_controller.py, body_tracker_opencv.py, api_server.py, osc_server.py, utils.py -->

- [x] Customize the Project
	<!--
	All core functionality has been implemented:
	- Real-time body tracking with OpenCV HOG detector (MediaPipe alternative for Python 3.13 compatibility)
	- ONVIF PTZ camera control
	- Multiple person detection and tracking
	- Smooth pan movements with dead zone
	- OSC and REST API interfaces
	- Command line controls
	- Bitfocus Companion integration ready
	- Control scripts for Windows (batch and PowerShell)
	- Comprehensive documentation and setup guide
	-->

- [x] Install Required Extensions
	<!-- No specific extensions required for Python project. -->

- [x] Compile the Project
	<!--
	Dependencies installed successfully:
	- OpenCV for computer vision and body detection
	- Flask for REST API server
	- python-osc for OSC server
	- onvif-zeep for camera control
	- All other required packages
	Note: MediaPipe not available for Python 3.13, using OpenCV HOG detector instead
	-->

- [x] Create and Run Task
	<!--
	Created VS Code tasks:
	- "PTZ Tracker - Start Daemon" - Runs the system in background with API/OSC servers
	- "PTZ Tracker - Run Tests" - Runs the test suite
	-->

- [x] Launch the Project
	<!--
	Project is ready to launch. Main application works correctly.
	Multiple ways to start:
	1. Command line: python main.py --daemon
	2. Windows batch: ptz_control.bat  
	3. PowerShell: ptz_control.ps1
	4. VS Code task: "PTZ Tracker - Start Daemon"
	-->

- [x] Ensure Documentation is Complete
	<!--
	Documentation complete:
	- README.md with comprehensive usage instructions
	- SETUP.md with detailed configuration guide
	- config.json with all settings documented
	- Control scripts for easy Windows usage
	- Test system for functionality verification
	-->