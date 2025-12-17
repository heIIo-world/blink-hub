# Blink Hub

A modern web-based application for managing and archiving videos from your Blink camera system.

![Version](https://img.shields.io/badge/version-3.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Features

### 📹 Video Management
- **Download videos** from all your Blink cameras
- **Browse & filter** by camera, date range, tags, and more
- **Calendar view** to quickly find videos by date
- **Bulk operations** - select multiple videos to tag, export as zip, or delete
- **Video notes** - add personal notes to any video
- **Star/favorite** important clips

### 📷 Camera Monitoring
- **Live camera status** with thumbnails
- **Sync module health** monitoring
- **WiFi signal strength** indicators
- **Battery/power status** for each camera
- **Camera groups** for organization

### 🎬 Video Player
- **In-browser playback** with speed controls (0.5x - 2x)
- **Picture-in-Picture** mode
- **Keyboard shortcuts** for power users
- **Direct download** button

### 📊 Dashboard
- **At-a-glance stats** - camera count, video count, storage used
- **Cloud sync status** - see what's not yet downloaded
- **Download missing videos** with one click
- **Activity overview** by camera

### ⚙️ Additional Features
- **Automatic sync** - schedule downloads on interval
- **Timezone support** - filenames in your local time
- **Tag system** - organize videos with custom tags
- **Dark/Light theme** - easy on the eyes
- **Responsive design** - works on desktop and mobile

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (tested), macOS, or Linux
- ffmpeg (optional, for video thumbnails)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/blink-hub.git
   cd blink-hub
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```
   Or on Windows, double-click `start.bat`

4. **Open in browser**
   Navigate to `http://localhost:8080`

### Windows Service (Optional)

To run as a background service that starts automatically:
```bash
install-service.bat
```

## Usage

### First Login
1. Enter your Blink account email and password
2. Check "Remember me" to save credentials
3. Enter the 2FA PIN sent to your phone
4. You're in!

### Downloading Videos
1. Go to the **Downloads** tab
2. Select number of days to download
3. Click **Download Now**
4. Videos are saved to the `downloads` folder

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` / `K` | Play/Pause video |
| `J` | Rewind 10 seconds |
| `L` | Forward 10 seconds |
| `←` / `→` | Skip 5 seconds |
| `↑` / `↓` | Volume up/down |
| `M` | Mute/unmute |
| `F` | Fullscreen |
| `Escape` | Close modal |

### Calendar View

The calendar bar shows which days have recorded videos:
- Days with videos show a dot indicator
- Click any day to filter videos to that date
- Use arrow buttons to navigate between months

### Bulk Export

1. Click video thumbnails to select multiple videos
2. Click **Export Zip** in the bulk actions bar
3. Videos are packaged into a zip file organized by camera

## Configuration

Settings are available in the **Settings** tab:

| Section | Options |
|---------|---------|
| **Account** | Login credentials, logout |
| **Organization** | Tags and camera groups |
| **Storage & Files** | Download location, retention policy |
| **Automatic Sync** | Schedule automatic downloads |
| **Cloud Sync** | Compare local vs cloud videos |
| **Timezone** | Set filename timezone |

## Project Structure

```
blink-hub/
├── app.py              # Main FastAPI application
├── templates/
│   └── index.html      # Single-page web UI
├── static/             # Static assets
├── downloads/          # Downloaded videos (auto-created)
├── thumbnails/         # Video thumbnails (auto-created)
├── logs/               # Application logs (auto-created)
├── requirements.txt    # Python dependencies
├── start.bat           # Windows launcher
├── install-service.bat # Windows service installer
└── README.md
```

## Tech Stack

- **Backend**: Python 3.8+, FastAPI, SQLite
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Blink API**: [blinkpy](https://github.com/fronzbot/blinkpy) library

## API Endpoints

The application exposes a REST API at `http://localhost:8080/api/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Application status |
| `/api/cameras` | GET | List all cameras |
| `/api/local-videos` | GET | Browse downloaded videos |
| `/api/local-videos/day-counts` | GET | Video counts by day (calendar) |
| `/api/local-videos/export` | POST | Export videos as zip |
| `/api/sync-modules` | GET | List sync modules |
| `/api/download` | POST | Start video download |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is not affiliated with, endorsed by, or connected to Amazon or Blink. Use at your own risk. This application stores your Blink credentials locally on your machine.

## Acknowledgments

- [blinkpy](https://github.com/fronzbot/blinkpy) - Python library for Blink cameras
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for Python

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/YOUR_USERNAME/blink-hub/issues).
