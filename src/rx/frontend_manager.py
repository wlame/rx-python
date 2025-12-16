"""
Frontend manager for RX - handles downloading and updating frontend from GitHub releases.

This module manages the frontend viewer by:
1. Checking GitHub for the latest release
2. Downloading frontend if needed
3. Caching in ~/.cache/rx/frontend/
4. Serving static files
"""

import json
import logging
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx


logger = logging.getLogger(__name__)

# Configuration
GITHUB_REPO = 'wlame/rx-viewer'  # Change to your actual repo
GITHUB_API_BASE = 'https://api.github.com'
GITHUB_RELEASES_URL = f'{GITHUB_API_BASE}/repos/{GITHUB_REPO}/releases/latest'
FRONTEND_CACHE_DIR = Path.home() / '.cache' / 'rx' / 'frontend'
VERSION_FILE = FRONTEND_CACHE_DIR / 'version.json'
CHECK_INTERVAL = timedelta(hours=24)  # Check for updates every 24 hours
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


class FrontendManager:
    """Manages frontend downloads and updates."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        repo: str = GITHUB_REPO,
        check_interval: timedelta = CHECK_INTERVAL,
    ):
        """Initialize frontend manager.

        Args:
            cache_dir: Directory to cache frontend (default: ~/.cache/rx/frontend)
            repo: GitHub repository (owner/name)
            check_interval: How often to check for updates
        """
        self.cache_dir = cache_dir or FRONTEND_CACHE_DIR
        self.repo = repo
        self.check_interval = check_interval
        self.releases_url = f'{GITHUB_API_BASE}/repos/{repo}/releases/latest'
        self.version_file = self.cache_dir / 'version.json'

    def get_cached_metadata(self) -> CacheMetadata | None:
        """Get metadata about cached frontend.

        Returns:
            CacheMetadata if cache exists, None otherwise
        """
        if not self.version_file.exists():
            return None

        try:
            with open(self.version_file) as f:
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

        with open(self.version_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def should_check_for_updates(self) -> bool:
        """Check if we should query GitHub for updates.

        Returns:
            True if we should check, False if recently checked
        """
        metadata = self.get_cached_metadata()
        if not metadata:
            return True

        try:
            last_check = datetime.fromisoformat(metadata.last_check.replace('Z', '+00:00'))
            return datetime.now().astimezone() - last_check > self.check_interval
        except Exception:
            return True

    async def get_latest_release(self) -> dict | None:
        """Get latest release info from GitHub.

        Returns:
            Release data or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    self.releases_url,
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
            logger.warning(f'Timeout checking for frontend updates from {self.releases_url}')
            return None
        except Exception as e:
            logger.warning(f'Failed to check for frontend updates: {e}')
            return None

    async def download_frontend(self, release: dict) -> bool:
        """Download frontend from release.

        Args:
            release: GitHub release data

        Returns:
            True if successful, False otherwise
        """
        # Find dist.tar.gz asset
        assets = release.get('assets', [])
        dist_asset = None

        for asset in assets:
            if asset['name'] == 'dist.tar.gz':
                dist_asset = asset
                break

        if not dist_asset:
            logger.error('No dist.tar.gz asset found in release')
            return False

        download_url = dist_asset['browser_download_url']

        logger.info(f'Downloading frontend from {download_url}')

        try:
            # Download to temporary file
            temp_file = self.cache_dir / 'dist.tar.gz.tmp'
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
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

            # Extract to cache directory
            logger.info(f'Extracting to {self.cache_dir}')

            with tarfile.open(temp_file, 'r:gz') as tar:
                # Security: verify paths don't escape cache dir
                for member in tar.getmembers():
                    member_path = Path(self.cache_dir) / member.name
                    if not str(member_path.resolve()).startswith(str(self.cache_dir.resolve())):
                        logger.error(f'Unsafe path in archive: {member.name}')
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
                    version=release.get('tag_name', 'unknown').lstrip('v'),
                    build_date=release.get('published_at', ''),
                    commit='',
                )

            # Save metadata
            metadata = CacheMetadata(
                version=version,
                downloaded_at=datetime.now().astimezone().isoformat(),
                last_check=datetime.now().astimezone().isoformat(),
                release_url=release.get('html_url', ''),
            )
            self.save_metadata(metadata)

            logger.info(f'Frontend v{version.version} installed successfully')
            return True

        except Exception as e:
            logger.error(f'Failed to download frontend: {e}')
            return False

    def is_frontend_available(self) -> bool:
        """Check if frontend is available in cache.

        Returns:
            True if frontend is cached and usable
        """
        index_html = self.cache_dir / 'index.html'
        assets_dir = self.cache_dir / 'assets'

        return index_html.exists() and assets_dir.exists() and assets_dir.is_dir()

    async def ensure_frontend(self, force_check: bool = False) -> bool:
        """Ensure frontend is available, download if needed.

        Args:
            force_check: Force check for updates even if recently checked

        Returns:
            True if frontend is available, False otherwise
        """
        # If frontend exists and we shouldn't check for updates, we're done
        if self.is_frontend_available() and not force_check and not self.should_check_for_updates():
            logger.debug('Frontend is cached and up to date')
            return True

        # Check for updates
        logger.info('Checking for frontend updates...')

        release = await self.get_latest_release()
        if not release:
            # Failed to check - use cached if available
            if self.is_frontend_available():
                logger.warning('Failed to check for updates, using cached frontend')
                return True
            else:
                logger.error('No cached frontend and failed to download')
                return False

        release_tag = release.get('tag_name', '').lstrip('v')
        cached_metadata = self.get_cached_metadata()

        # Update last_check time
        if cached_metadata:
            cached_metadata.last_check = datetime.now().astimezone().isoformat()
            self.save_metadata(cached_metadata)

        # Check if we need to download
        if cached_metadata and cached_metadata.version.version == release_tag:
            logger.info(f'Frontend v{release_tag} is up to date')
            return True

        # Download new version
        if cached_metadata:
            logger.info(f'Updating frontend from v{cached_metadata.version.version} to v{release_tag}')
        else:
            logger.info(f'Downloading frontend v{release_tag}')

        return await self.download_frontend(release)

    def get_frontend_dir(self) -> Path:
        """Get frontend directory path.

        Returns:
            Path to frontend directory
        """
        return self.cache_dir


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


def get_frontend_dir() -> Path:
    """Get frontend directory (convenience function).

    Returns:
        Path to frontend directory
    """
    manager = get_frontend_manager()
    return manager.get_frontend_dir()
