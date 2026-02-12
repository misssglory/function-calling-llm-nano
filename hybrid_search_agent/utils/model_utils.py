"""Model download and verification utilities"""

import os
import sys
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from loguru import logger
import hashlib
from tqdm import tqdm
from datetime import datetime

from hybrid_search_agent.config import AVAILABLE_MODELS, MODELS_DIR


class ModelManager:
    """Manages model downloads and local model information"""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def get_downloaded_models(self) -> Dict[str, Dict]:
        """Get list of downloaded models with their info"""
        downloaded = {}
        
        # Get all .gguf files in models directory
        model_files = list(self.models_dir.glob("*.gguf"))
        
        for model_file in model_files:
            file_size = model_file.stat().st_size
            modified_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            
            # Find matching model in catalog
            model_key = None
            model_info = None
            for key, info in AVAILABLE_MODELS.items():
                if info["filename"] == model_file.name:
                    model_key = key
                    model_info = info.copy()
                    break
            
            if not model_info:
                # Model not in catalog, create basic info
                model_info = {
                    "filename": model_file.name,
                    "description": "Custom model",
                    "language": "unknown",
                    "size_gb": file_size / (1024**3),
                    "context": 4096,
                    "downloaded": True,
                    "file_size": file_size,
                    "modified": modified_time,
                    "path": str(model_file)
                }
            else:
                model_info["downloaded"] = True
                model_info["file_size"] = file_size
                model_info["modified"] = modified_time
                model_info["path"] = str(model_file)
            
            downloaded[model_key or model_file.stem] = model_info
        
        return downloaded
    
    def is_downloaded(self, model_name_or_key: str) -> bool:
        """Check if model is already downloaded"""
        downloaded = self.get_downloaded_models()
        
        # Check by key
        if model_name_or_key in downloaded:
            return True
        
        # Check by filename
        for key, info in downloaded.items():
            if isinstance(info, dict) and info.get("filename") == model_name_or_key:
                return True
            if key == model_name_or_key:
                return True
        
        return False
    
    def get_model_path(self, model_name_or_key: str) -> Optional[str]:
        """Get full path to downloaded model"""
        downloaded = self.get_downloaded_models()
        
        # Check by key
        if model_name_or_key in downloaded:
            return downloaded[model_name_or_key].get("path")
        
        # Check by filename
        for key, info in downloaded.items():
            if info.get("filename") == model_name_or_key:
                return info.get("path")
        
        return None
    
    def verify_model_file(self, model_path: str) -> bool:
        """Verify that model file exists and is not corrupted"""
        path = Path(model_path)
        
        if not path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Check file size (should be at least 100MB)
        file_size = path.stat().st_size
        if file_size < 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Model file is suspiciously small: {file_size / 1024 / 1024:.1f}MB")
            return False
        
        # Try to read first few bytes
        try:
            with open(path, 'rb') as f:
                header = f.read(4)
                # GGUF files start with 'GGUF' magic number
                if header != b'GGUF':
                    logger.warning(f"File does not appear to be a valid GGUF model (magic: {header})")
        except Exception as e:
            logger.error(f"Cannot read model file: {e}")
            return False
        
        logger.debug(f"Model file verified: {path.name} ({file_size / 1024 / 1024:.1f}MB)")
        return True
    
    def download_model(self, model_key: str) -> Optional[str]:
        """Download GGUF model from Hugging Face"""
        
        if model_key not in AVAILABLE_MODELS:
            logger.error(f"Unknown model key: {model_key}")
            self.list_available_models()
            return None
        
        model_info = AVAILABLE_MODELS[model_key]
        filename = model_info["filename"]
        output_path = self.models_dir / filename
        
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            if self.verify_model_file(str(output_path)):
                return str(output_path)
            else:
                logger.warning(f"Existing model file is corrupted, re-downloading...")
                output_path.unlink()
        
        logger.info(f"Downloading {filename} ({model_info['description']})")
        logger.info(f"Size: {model_info['size_gb']}GB")
        logger.info(f"Language: {model_info['language']}")
        
        try:
            # Download with progress bar
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=filename, ncols=80) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.success(f"Model downloaded successfully: {output_path}")
            
            # Verify downloaded file
            if self.verify_model_file(str(output_path)):
                return str(output_path)
            else:
                logger.error("Downloaded file is corrupted")
                output_path.unlink()
                return None
                
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if output_path.exists():
                output_path.unlink()
            return None
    
    def resolve_model_path(self, model_name_or_path: str, auto_download: bool = True) -> str:
        """Resolve model path from name/key or full path"""
        
        # Check if it's a full path that exists
        if Path(model_name_or_path).exists():
            if self.verify_model_file(model_name_or_path):
                return model_name_or_path
            else:
                logger.warning(f"Model file exists but may be corrupted: {model_name_or_path}")
                return model_name_or_path
        
        # Check if it's a model key or filename in downloaded models
        model_path = self.get_model_path(model_name_or_path)
        if model_path:
            return model_path
        
        # Check if it's a model key in available models
        if model_name_or_path in AVAILABLE_MODELS:
            if auto_download:
                logger.info(f"Model '{model_name_or_path}' not downloaded. Starting download...")
                downloaded_path = self.download_model(model_name_or_path)
                if downloaded_path:
                    return downloaded_path
            else:
                raise FileNotFoundError(
                    f"Model '{model_name_or_path}' not downloaded and auto_download=False"
                )
        
        # Try to find by partial match in downloaded models
        downloaded = self.get_downloaded_models()
        for key, info in downloaded.items():
            if model_name_or_path.lower() in key.lower() or \
               model_name_or_path.lower() in info.get('filename', '').lower():
                logger.info(f"Found matching model: {key}")
                return info.get('path')
        
        # If nothing found, try to download default model
        if auto_download:
            logger.warning(f"Model '{model_name_or_path}' not found. Trying default model...")
            default_model = self._get_default_model()
            if default_model:
                logger.info(f"Downloading default model: {default_model}")
                return self.resolve_model_path(default_model, auto_download=True)
        
        raise FileNotFoundError(
            f"Model not found: {model_name_or_path}\n"
            f"Use list_downloaded_models() to see available models\n"
            f"Or download with: download_model('model_key')"
        )
    
    def _get_default_model(self) -> Optional[str]:
        """Get default model key"""
        for key, info in AVAILABLE_MODELS.items():
            if info.get('default', False):
                return key
        return "tinyllama"  # Fallback default
    
    def list_downloaded_models(self) -> List[Dict]:
        """List only downloaded models with details"""
        downloaded = self.get_downloaded_models()
        
        if not downloaded:
            print("\n" + "="*60)
            print("📭 NO DOWNLOADED MODELS")
            print("="*60)
            print(f"\nModels directory: {self.models_dir}")
            print("\n💡 Download a model:")
            print("   from hybrid_search_agent.utils.model_utils import download_model")
            print("   download_model('tinyllama')")
            print("="*60)
            return []
        
        print("\n" + "="*80)
        print(f"📦 DOWNLOADED MODELS ({len(downloaded)})")
        print("="*80)
        
        model_list = []
        for key, info in downloaded.items():
            size_gb = info.get('file_size', 0) / (1024**3)
            modified = info.get('modified', datetime.now()).strftime('%Y-%m-%d %H:%M')
            
            model_entry = {
                "key": key,
                "filename": info.get('filename', 'Unknown'),
                "description": info.get('description', 'Custom model'),
                "language": info.get('language', 'unknown'),
                "size_gb": size_gb,
                "modified": modified,
                "path": info.get('path', ''),
                "context": info.get('context', 4096)
            }
            model_list.append(model_entry)
            
            # Display model info
            print(f"\n📌 {key}")
            print(f"   📝 {info.get('description', 'Custom model')}")
            print(f"   📄 File: {info.get('filename', 'Unknown')}")
            print(f"   🌐 Language: {info.get('language', 'unknown').upper()}")
            print(f"   💾 Size: {size_gb:.1f}GB")
            print(f"   📅 Downloaded: {modified}")
        
        print("\n" + "="*80)
        return model_list
    
    def list_available_models(self, language: Optional[str] = None):
        """List all available models for download"""
        downloaded = self.get_downloaded_models()
        
        print("\n" + "="*80)
        print("📋 AVAILABLE MODELS FOR DOWNLOAD")
        print("="*80)
        
        for key, info in AVAILABLE_MODELS.items():
            # Filter by language if specified
            if language and info['language'] != language:
                continue
            
            is_downloaded = self.is_downloaded(key) or self.is_downloaded(info['filename'])
            status = "✅ DOWNLOADED" if is_downloaded else "⬇️  AVAILABLE"
            
            print(f"\n{status}")
            print(f"   Key: {key}")
            print(f"   📝 {info['description']}")
            print(f"   📄 File: {info['filename']}")
            print(f"   🌐 Language: {info['language'].upper()}")
            print(f"   💾 Size: {info['size_gb']}GB")
        
        print("\n" + "="*80)
        print("To download a model:")
        print("  from hybrid_search_agent.utils.model_utils import download_model")
        print("  download_model('tinyllama')")
        print("="*80)


# Global instance for easy import
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create global ModelManager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Convenience functions
def get_downloaded_models() -> Dict[str, Dict]:
    """Get list of downloaded models"""
    return get_model_manager().get_downloaded_models()

def list_downloaded_models() -> List[Dict]:
    """List only downloaded models"""
    return get_model_manager().list_downloaded_models()

def list_available_models(language: Optional[str] = None):
    """List all available models for download"""
    get_model_manager().list_available_models(language)

def download_model(model_key: str) -> Optional[str]:
    """Download a model by key"""
    return get_model_manager().download_model(model_key)

def resolve_model_path(model_name_or_path: str, auto_download: bool = True) -> str:
    """Resolve model path from name/key"""
    return get_model_manager().resolve_model_path(model_name_or_path, auto_download)

def verify_model_file(model_path: str) -> bool:
    """Verify model file"""
    return get_model_manager().verify_model_file(model_path)