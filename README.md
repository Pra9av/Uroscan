🏥 UroScan – AI-Powered Kidney Stone Detection Platform
📌 Overview

UroScan is a full-stack AI-powered clinical web application designed for automated kidney stone detection from CT scan images.
It integrates deep learning segmentation with a structured hospital-style review workflow for technicians and doctors.

The platform simulates a real-world medical imaging system with role-based access, AI inference, and diagnostic visualization.

🚀 Features
👨‍🔬 Technician Dashboard

Upload CT scan images

Trigger AI analysis

View segmentation results

View heatmap confidence map

Automatically assign scan to doctor

👨‍⚕️ Doctor Dashboard

View assigned scans

Side-by-side Original CT & AI Segmentation

Confidence Heatmap visualization

Severity classification (Mild / Moderate / High)

Add clinical notes

Mark scan as Reviewed (status tracking)

🤖 AI Model

U-Net (ResNet34 encoder)

Kidney & Stone segmentation

Pixel-level classification

Confidence-based severity detection

Heatmap visualization using OpenCV

🏗 Architecture

Frontend → Angular
Backend → FastAPI
Database → MongoDB + GridFS
AI Model → PyTorch (Segmentation Models PyTorch)
