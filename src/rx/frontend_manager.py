"""
Frontend manager for RX - handles downloading and updating frontend from GitHub releases.

This module manages the frontend viewer by:
1. Checking GitHub for the latest release
2. Downloading frontend if needed
3. Caching in ~/.cache/rx/frontend/ (or custom path)
4. Serving static files securely

Environment Variables:
- RX_FRONTEND_VERSION: Override version to fetch (e.g., "v1.2.12" or "latest")
- RX_FRONTEND_URL: Override URL to download dist.tar.gz (forces fetch)
- RX_FRONTEND_PATH: Override cache directory path
"""

import json
import logging
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx


logger = logging.getLogger(__name__)

# Configuration
GITHUB_REPO = 'wlame/rx-viewer'
GITHUB_API_BASE = 'https://api.github.com'
DEFAULT_FRONTEND_CACHE_DIR = Path.home() / '.cache' / 'rx' / 'frontend'
REQUEST_TIMEOUT = 30.0  # seconds


@dataclass
class FrontendVersion:
    """Frontend version information."""

    version: str
    build_date: str
    commit: str

    @classmethod
    def from_dict(cls, data: dict) -> 'FrontendVersion':
        """Create from dictionary."""
        return cls(
            version=data.get('version', 'unknown'),
            build_date=data.get('buildDate', ''),
            commit=data.get('commit', ''),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'buildDate': self.build_date,
            'commit': self.commit,
        }


@dataclass
class CacheMetadata:
    """Metadata about cached frontend."""

    version: FrontendVersion
    downloaded_at: str
    last_check: str
    release_url: str

    @classmethod
    def from_dict(cls, data: dict) -> 'CacheMetadata':
        """Create from dictionary."""
        return cls(
            version=FrontendVersion.from_dict(data.get('version', {})),
            downloaded_at=data.get('downloaded_at', ''),
            last_check=data.get('last_check', ''),
            release_url=data.get('release_url', ''),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'version': self.version.to_dict(),
            'downloaded_at': self.downloaded_at,
            'last_check': self.last_check,
            'release_url': self.release_url,
        }


def validate_path_security(base_path: Path, requested_path: Path) -> bool:
    """Validate that requested path is within base path (prevent directory traversal).

    Args:
        base_path: Base directory (e.g., frontend cache)
        requested_path: Path to validate

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve both paths to absolute
        base_resolved = base_path.resolve()
        requested_resolved = requested_path.resolve()

        # Check if requested path starts with base path
        return str(requested_resolved).startswith(str(base_resolved))
    except Exception as e:
        logger.error(f'Path validation error: {e}')
        return False


class FrontendManager:
    """Manages frontend downloads and updates."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        repo: str = GITHUB_REPO,
    ):
        """Initialize frontend manager.

        Args:
            cache_dir: Directory to cache frontend (default: from RX_FRONTEND_PATH or ~/.cache/rx/frontend)
            repo: GitHub repository (owner/name)
        """
        # Read environment variables
        env_frontend_path = os.getenv('RX_FRONTEND_PATH')
        env_frontend_version = os.getenv('RX_FRONTEND_VERSION')
        env_frontend_url = os.getenv('RX_FRONTEND_URL')

        # Set cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        elif env_frontend_path:
            self.cache_dir = Path(env_frontend_path).expanduser()
            logger.info(f'Using frontend path from RX_FRONTEND_PATH: {self.cache_dir}')
        else:
            self.cache_dir = DEFAULT_FRONTEND_CACHE_DIR

        self.repo = repo
        self.version_file = self.cache_dir / 'version.json'
        self.metadata_file = self.cache_dir / '.metadata.json'

        # Store env vars
        self.env_version = env_frontend_version
        self.env_url = env_frontend_url

    def get_cached_metadata(self) -> CacheMetadata | None:
        """Get metadata about cached frontend.

        Returns:
            CacheMetadata if cache exists, None otherwise
        """
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f'Failed to read cache metadata: {e}')
            return None

    def save_metadata(self, metadata: CacheMetadata):
        """Save cache metadata.

        Args:
            metadata: Metadata to save
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    async def get_latest_release_info(self) -> dict | None:
        """Get latest release info from GitHub API.

        Returns:
            Release data or None if failed
        """
        url = f'{GITHUB_API_BASE}/repos/{self.repo}/releases/latest'

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={'Accept': 'application/vnd.github+json'},
                )

                if response.status_code == 404:
                    logger.warning(f'GitHub repository {self.repo} not found or has no releases')
                    return None

                if response.status_code != 200:
                    logger.warning(f'GitHub API returned {response.status_code}')
                    return None

                return response.json()
        except httpx.TimeoutException:
            logger.warning(f'Timeout checking for frontend updates from {url}')
            return None
        except Exception as e:
            logger.warning(f'Failed to check for frontend updates: {e}')
            return None

    async def get_version_release_info(self, version: str) -> dict | None:
        """Get specific version release info from GitHub API.

        Args:
            version: Version tag (e.g., "v1.2.12")

        Returns:
            Release data or None if failed
        """
        # Ensure version has 'v' prefix
        if not version.startswith('v'):
            version = f'v{version}'

        url = f'{GITHUB_API_BASE}/repos/{self.repo}/releases/tags/{version}'

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={'Accept': 'application/vnd.github+json'},
                )

                if response.status_code == 404:
                    logger.warning(f'Release {version} not found in {self.repo}')
                    return None

                if response.status_code != 200:
                    logger.warning(f'GitHub API returned {response.status_code}')
                    return None

                return response.json()
        except httpx.TimeoutException:
            logger.warning(f'Timeout checking for version {version}')
            return None
        except Exception as e:
            logger.warning(f'Failed to check for version {version}: {e}')
            return None

    def get_download_url_from_release(self, release: dict) -> str | None:
        """Extract dist.tar.gz download URL from release data.

        Args:
            release: GitHub release data

        Returns:
            Download URL or None if not found
        """
        assets = release.get('assets', [])
        for asset in assets:
            if asset['name'] == 'dist.tar.gz':
                return asset['browser_download_url']

        logger.error('No dist.tar.gz asset found in release')
        return None

    def get_direct_download_url(self, version: str) -> str:
        """Construct direct download URL for a specific version.

        Args:
            version: Version tag (e.g., "v1.2.12" or "latest")

        Returns:
            Direct download URL
        """
        if version == 'latest':
            return f'https://github.com/{self.repo}/releases/latest/download/dist.tar.gz'
        else:
            # Ensure version has 'v' prefix
            if not version.startswith('v'):
                version = f'v{version}'
            return f'https://github.com/{self.repo}/releases/download/{version}/dist.tar.gz'

    async def download_frontend(self, download_url: str, release_tag: str = 'unknown') -> bool:
        """Download frontend from URL.

        Args:
            download_url: URL to download dist.tar.gz from
            release_tag: Release tag for metadata

        Returns:
            True if successful, False otherwise
        """
        logger.info(f'Downloading frontend from {download_url}')

        try:
            # Download to temporary file
            temp_file = self.cache_dir / 'dist.tar.gz.tmp'
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
                async with client.stream('GET', download_url) as response:
                    if response.status_code != 200:
                        logger.error(f'Download failed with status {response.status_code}')
                        return False

                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0

                    with open(temp_file, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0 and downloaded % (100 * 1024) == 0:
                                progress = (downloaded / total_size) * 100
                                logger.debug(f'Download progress: {progress:.1f}%')

            logger.info(f'Downloaded {downloaded} bytes')

            # Clear existing frontend files (but keep metadata for now)
            for item in self.cache_dir.iterdir():
                if item.name not in ['dist.tar.gz.tmp', '.metadata.json']:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

            # Extract to cache directory
            logger.info(f'Extracting to {self.cache_dir}')

            with tarfile.open(temp_file, 'r:gz') as tar:
                # Security: verify paths don't escape cache dir
                for member in tar.getmembers():
                    member_path = Path(self.cache_dir) / member.name
                    if not validate_path_security(self.cache_dir, member_path):
                        logger.error(f'Unsafe path in archive: {member.name}')
                        temp_file.unlink()
                        return False

                tar.extractall(self.cache_dir)

            # Remove temp file
            temp_file.unlink()

            # Read version from extracted version.json
            version_json = self.cache_dir / 'version.json'
            if version_json.exists():
                with open(version_json) as f:
                    version_data = json.load(f)
                version = FrontendVersion.from_dict(version_data)
            else:
                # Fallback to release tag
                version = FrontendVersion(
                    version=release_tag.lstrip('v'),
                    build_date=datetime.now().isoformat(),
                    commit='',
                )

            # Save metadata
            metadata = CacheMetadata(
                version=version,
                downloaded_at=datetime.now().isoformat(),
                last_check=datetime.now().isoformat(),
                release_url=download_url,
            )
            self.save_metadata(metadata)

            logger.info(f'Frontend v{version.version} installed successfully')
            return True

        except Exception as e:
            logger.error(f'Failed to download frontend: {e}')
            if temp_file.exists():
                temp_file.unlink()
            return False

    def is_frontend_available(self) -> bool:
        """Check if frontend is available in cache.

        Returns:
            True if frontend is cached and usable
        """
        index_html = self.cache_dir / 'index.html'
        assets_dir = self.cache_dir / 'assets'

        return index_html.exists() and assets_dir.exists() and assets_dir.is_dir()

    async def ensure_frontend(self) -> bool:
        """Ensure frontend is available, download if needed.

        Logic:
        1. If RX_FRONTEND_URL is set: always download from that URL
        2. If RX_FRONTEND_VERSION is set: download that version if not cached
        3. If local version exists and no env vars: use cached (no GitHub requests)
        4. If no local version: download latest

        Returns:
            True if frontend is available, False otherwise
        """
        # Case 1: RX_FRONTEND_URL is set - force download
        if self.env_url:
            logger.info(f'RX_FRONTEND_URL is set, downloading from: {self.env_url}')
            return await self.download_frontend(self.env_url, 'custom')

        # Case 2: RX_FRONTEND_VERSION is set
        if self.env_version:
            cached_metadata = self.get_cached_metadata()

            # Normalize versions for comparison
            requested_version = self.env_version.lstrip('v')
            cached_version = cached_metadata.version.version.lstrip('v') if cached_metadata else None

            # If we already have the requested version cached, use it
            if cached_version and cached_version == requested_version:
                logger.info(f'Frontend v{requested_version} is already cached')
                return self.is_frontend_available()

            # Download the requested version
            if self.env_version.lower() == 'latest':
                logger.info('RX_FRONTEND_VERSION=latest, checking GitHub for latest release')
                release = await self.get_latest_release_info()
                if release:
                    download_url = self.get_download_url_from_release(release)
                    if download_url:
                        return await self.download_frontend(download_url, release.get('tag_name', 'latest'))
            else:
                logger.info(f'RX_FRONTEND_VERSION={self.env_version}, downloading specific version')
                download_url = self.get_direct_download_url(self.env_version)
                return await self.download_frontend(download_url, self.env_version)

            # Failed to download requested version
            if self.is_frontend_available():
                logger.warning('Failed to download requested version, using cached frontend')
                return True
            return False

        # Case 3: No env vars - use cached if available (no GitHub requests)
        if self.is_frontend_available():
            logger.debug('Frontend is cached, no env vars set - using cached version')
            return True

        # Case 4: No cache, no env vars - download latest
        logger.info('No cached frontend found, downloading latest release')
        release = await self.get_latest_release_info()
        if not release:
            logger.error('Failed to fetch latest release and no cache available')
            return False

        download_url = self.get_download_url_from_release(release)
        if not download_url:
            return False

        return await self.download_frontend(download_url, release.get('tag_name', 'latest'))

    def get_frontend_dir(self) -> Path:
        """Get frontend directory path.

        Returns:
            Path to frontend directory
        """
        return self.cache_dir

    def validate_static_file_path(self, requested_path: str) -> Path | None:
        """Validate and resolve a static file path.

        Security: Prevents directory traversal attacks (e.g., ../../etc/passwd)

        Args:
            requested_path: Requested file path (relative to frontend dir)

        Returns:
            Resolved absolute path if valid, None if invalid/unsafe
        """
        try:
            # Construct full path
            full_path = self.cache_dir / requested_path

            # Validate it's within cache directory
            if not validate_path_security(self.cache_dir, full_path):
                logger.warning(f'Rejected unsafe path: {requested_path}')
                return None

            # Check file exists
            if not full_path.exists() or not full_path.is_file():
                return None

            return full_path
        except Exception as e:
            logger.error(f'Path validation error for {requested_path}: {e}')
            return None


# Global instance
_manager: FrontendManager | None = None


def get_frontend_manager() -> FrontendManager:
    """Get global frontend manager instance.

    Returns:
        FrontendManager instance
    """
    global _manager
    if _manager is None:
        _manager = FrontendManager()
    return _manager


async def ensure_frontend() -> bool:
    """Ensure frontend is available (convenience function).

    Returns:
        True if frontend is available
    """
    manager = get_frontend_manager()
    return await manager.ensure_frontend()


def get_frontend_dir() -> Path | None:
    """Get frontend directory (convenience function).

    Returns:
        Path to frontend directory if available, None otherwise
    """
    manager = get_frontend_manager()
    if manager.is_frontend_available():
        return manager.get_frontend_dir()
    return None


def validate_static_path(requested_path: str) -> Path | None:
    """Validate static file path (convenience function).

    Args:
        requested_path: Requested file path

    Returns:
        Validated path or None
    """
    manager = get_frontend_manager()
    return manager.validate_static_file_path(requested_path)
