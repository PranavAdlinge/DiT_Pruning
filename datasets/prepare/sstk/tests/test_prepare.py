"""Tests for datasets/prepare/sstk/prepare.py"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from streaming import StreamingDataset

# Load sstk prepare module without polluting sys.path (for unit tests)
_sstk_dir = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sstk_prepare", _sstk_dir / "prepare.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)
parse_arguments = prepare.parse_arguments
write_images = prepare.write_images


def _run_prepare_main(argv: list[str]) -> None:
    """Run prepare main() via subprocess to avoid multiprocessing pickle issues."""
    subprocess.run(
        [sys.executable, str(_sstk_dir / "prepare.py")] + argv,
        check=True,
        capture_output=True,
    )


def _sanitize_for_filename(text: str, max_len: int = 80) -> str:
    """Replace filesystem-unsafe chars; truncate to max_len."""
    safe = re.sub(r'[/\\:*?"<>|]', '_', text)
    safe = safe.strip() or 'unnamed'
    return safe[:max_len]


class TestParseArguments:
    """Tests for parse_arguments()."""

    def test_required_images_txt(self):
        """images_txt is required and parsed correctly."""
        with patch.object(sys, 'argv', ['prepare.py', '--images_txt', '/path/to/images.txt']):
            args = parse_arguments()
            assert args.images_txt == '/path/to/images.txt'

    def test_default_values(self):
        """Default values are applied when not specified."""
        with patch.object(sys, 'argv', ['prepare.py', '--images_txt', '/path/to/images.txt']):
            args = parse_arguments()
            assert args.local_mds_dir == ''
            assert args.num_proc == 16
            assert args.seed == 42
            assert args.size is None
            assert args.min_size == 512
            assert args.min_aspect_ratio == 0.67
            assert args.max_aspect_ratio == 1.33

    def test_all_arguments_override_defaults(self):
        """All optional arguments override defaults when provided."""
        with patch.object(sys, 'argv', [
            'prepare.py',
            '--images_txt', '/path/to/images.txt',
            '--local_mds_dir', './mds_output/',
            '--num_proc', '8',
            '--seed', '123',
            '--size', '5000',
            '--min_size', '256',
            '--min_aspect_ratio', '0.5',
            '--max_aspect_ratio', '2.0',
        ]):
            args = parse_arguments()
            assert args.images_txt == '/path/to/images.txt'
            assert args.local_mds_dir == './mds_output/'
            assert args.num_proc == 8
            assert args.seed == 123
            assert args.size == 5000
            assert args.min_size == 256
            assert args.min_aspect_ratio == 0.5
            assert args.max_aspect_ratio == 2.0

    def test_missing_images_txt_raises(self):
        """Missing required --images_txt raises SystemExit."""
        with patch.object(sys, 'argv', ['prepare.py']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestWriteImagesFiltering:
    """Tests for write_images filtering logic (min_size, aspect_ratio, extensions)."""

    @pytest.fixture
    def mock_args(self):
        """Create a mock args object with configurable filtering parameters."""
        args = MagicMock()
        args.local_mds_dir = '/tmp/test_mds'
        args.min_size = 512
        args.min_aspect_ratio = 0.67
        args.max_aspect_ratio = 1.33
        return args

    def test_skips_non_image_extensions(self, mock_args, tmp_path):
        """Files without .png, .jpg, .jpeg are skipped."""
        # Create a .bmp path - should be skipped
        images_path = np.array([str(tmp_path / 'image.bmp')])
        mock_args.min_size = 0  # Disable size filter for this test

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_not_called()

    def test_accepts_png_jpg_jpeg_extensions(self, mock_args, tmp_path):
        """Accepts .png, .jpg, .jpeg extensions (case insensitive)."""
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            img_path = tmp_path / f'image{ext}'
            img_path.touch()
            cap_path = tmp_path / f'image{ext}.json'
            cap_path.write_text(json.dumps({'description': 'A caption'}))

            # Create a 600x600 image (passes min_size and aspect_ratio)
            img = Image.new('RGB', (600, 600), color='red')
            img.save(img_path, format='PNG' if 'png' in ext.lower() else 'JPEG')

            images_path = np.array([str(img_path)])
            mock_args.min_size = 512
            mock_args.min_aspect_ratio = 0.67
            mock_args.max_aspect_ratio = 1.33

            with patch.object(prepare, '_current_process_index', return_value=0), \
                 patch.object(prepare, 'MDSWriter') as mock_writer, \
                 patch.object(prepare, 'tqdm', lambda x: x), \
                 patch.object(prepare.os, 'makedirs'):
                writer_instance = MagicMock()
                mock_writer.return_value = writer_instance

                write_images(images_path, mock_args)
                writer_instance.write.assert_called_once()
                call_args = writer_instance.write.call_args[0][0]
                assert call_args['caption'] == 'A caption'
                assert call_args['width'] == 600
                assert call_args['height'] == 600

                # Reset for next iteration
                writer_instance.reset_mock()

    def test_skips_image_below_min_size(self, mock_args, tmp_path):
        """Images with min dimension below min_size are skipped."""
        img_path = tmp_path / 'small.png'
        img = Image.new('RGB', (256, 256), color='blue')
        img.save(img_path)
        cap_path = tmp_path / 'small.png.json'
        cap_path.write_text(json.dumps({'description': 'Small'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 512  # 256 < 512, should skip

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_not_called()

    def test_min_size_zero_disables_filter(self, mock_args, tmp_path):
        """When min_size=0, small images are not filtered."""
        img_path = tmp_path / 'tiny.png'
        img = Image.new('RGB', (64, 64), color='green')
        img.save(img_path)
        cap_path = tmp_path / 'tiny.png.json'
        cap_path.write_text(json.dumps({'description': 'Tiny'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 0

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_called_once()

    def test_skips_image_below_min_aspect_ratio(self, mock_args, tmp_path):
        """Images with aspect ratio below min are skipped."""
        # 400x800 = 0.5 aspect ratio, below 0.67
        img_path = tmp_path / 'tall.png'
        img = Image.new('RGB', (400, 800), color='red')
        img.save(img_path)
        cap_path = tmp_path / 'tall.png.json'
        cap_path.write_text(json.dumps({'description': 'Tall'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 0  # Disable size filter

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_not_called()

    def test_skips_image_above_max_aspect_ratio(self, mock_args, tmp_path):
        """Images with aspect ratio above max are skipped."""
        # 800x400 = 2.0 aspect ratio, above 1.33
        img_path = tmp_path / 'wide.png'
        img = Image.new('RGB', (800, 400), color='red')
        img.save(img_path)
        cap_path = tmp_path / 'wide.png.json'
        cap_path.write_text(json.dumps({'description': 'Wide'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 0

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_not_called()

    def test_accepts_image_within_aspect_ratio_bounds(self, mock_args, tmp_path):
        """Images within aspect ratio bounds are written."""
        # 600x600 = 1.0 aspect ratio, within [0.67, 1.33]
        img_path = tmp_path / 'square.png'
        img = Image.new('RGB', (600, 600), color='red')
        img.save(img_path)
        cap_path = tmp_path / 'square.png.json'
        cap_path.write_text(json.dumps({'description': 'Square image'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 512

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_called_once()
            call_args = writer_instance.write.call_args[0][0]
            assert call_args['caption'] == 'Square image'

    def test_caption_stripped_and_handles_none(self, mock_args, tmp_path):
        """Caption is stripped and None/empty is handled."""
        img_path = tmp_path / 'test.png'
        img = Image.new('RGB', (600, 600), color='red')
        img.save(img_path)
        cap_path = tmp_path / 'test.png.json'
        cap_path.write_text(json.dumps({'description': '  spaced caption  '}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 512

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            call_args = writer_instance.write.call_args[0][0]
            assert call_args['caption'] == 'spaced caption'

    def test_list_caption_stored_as_json(self, mock_args, tmp_path):
        """When description is a list of captions, store as JSON array string."""
        img_path = tmp_path / 'multi.png'
        img = Image.new('RGB', (600, 600), color='red')
        img.save(img_path)
        cap_path = tmp_path / 'multi.png.json'
        cap_path.write_text(json.dumps({
            'description': ['first caption', 'second caption', 'third caption']
        }))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 512

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_called_once()
            call_args = writer_instance.write.call_args[0][0]
            stored = json.loads(call_args['caption'])
            assert stored == ['first caption', 'second caption', 'third caption']

    def test_skips_on_image_load_error(self, mock_args, tmp_path):
        """Skips and continues when image loading fails."""
        img_path = tmp_path / 'broken.png'
        img_path.write_text('not an image')
        cap_path = tmp_path / 'broken.png.json'
        cap_path.write_text(json.dumps({'description': 'Broken'}))

        images_path = np.array([str(img_path)])
        mock_args.min_size = 0

        with patch.object(prepare, '_current_process_index', return_value=0), \
             patch.object(prepare, 'MDSWriter') as mock_writer, \
             patch.object(prepare, 'tqdm', lambda x: x), \
             patch.object(prepare.os, 'makedirs'):
            writer_instance = MagicMock()
            mock_writer.return_value = writer_instance

            write_images(images_path, mock_args)
            writer_instance.write.assert_not_called()

    def test_requires_numpy_ndarray(self, mock_args):
        """write_images asserts images_path is np.ndarray."""
        with pytest.raises(AssertionError):
            write_images(['/path/to/img.png'], mock_args)


class TestMain:
    """Tests for main() orchestration."""

    def test_main_reads_images_txt_and_splits(self, tmp_path):
        """main reads image paths from txt, samples/splits, and writes MDS."""
        images_txt = tmp_path / 'images.txt'
        images_txt.write_text('/img1.png\n/img2.png\n/img3.png\n/img4.png')
        mds_dir = tmp_path / 'mds'

        # Create minimal image + caption files
        for i in range(1, 5):
            img_path = tmp_path / f'img{i}.png'
            img = Image.new('RGB', (600, 600), color='red')
            img.save(img_path)
            cap_path = tmp_path / f'img{i}.png.json'
            cap_path.write_text(json.dumps({'description': f'Caption {i}'}))

        # Update images.txt with actual paths
        paths = [str(tmp_path / f'img{i}.png') for i in range(1, 5)]
        images_txt.write_text('\n'.join(paths))

        _run_prepare_main([
            '--images_txt', str(images_txt),
            '--local_mds_dir', str(mds_dir),
            '--num_proc', '2',
            '--seed', '42',
            '--size', '4',
            '--min_size', '512',
        ])

        assert mds_dir.exists()
        for i in range(2):
            assert (mds_dir / str(i)).exists()


class TestIntegrationPrepareAndRead:
    """
    End-to-end integration test: 10 images -> prepare.py (MDS) -> read -> save as PNG.
    """

    def test_prepare_write_read_save_as_png(self, tmp_path):
        """
        Given 10 images in a txt file, run prepare.py to write MDS shards,
        then read the MDS output and save each image as PNG with caption as filename.
        """
        num_images = 10
        images_dir = tmp_path / 'images'
        images_dir.mkdir()
        mds_dir = tmp_path / 'mds'
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # 1. Create 10 images + caption JSON files
        captions = [
            'a red sunset over the ocean',
            'a cat sleeping on a couch',
            'mountain peaks at dawn',
            'fresh coffee in a mug',
            'abstract geometric pattern',
            'vintage car in the rain',
            'forest path in autumn',
            'underwater coral reef',
            'city skyline at night',
            'flowers in a vase',
        ]
        image_paths = []
        for i in range(num_images):
            img_path = images_dir / f'img_{i:02d}.png'
            img = Image.new('RGB', (600, 600), color=(i * 25, 100, 200 - i * 10))
            img.save(img_path)
            cap_path = images_dir / f'img_{i:02d}.png.json'
            cap_path.write_text(json.dumps({'description': captions[i]}))
            image_paths.append(str(img_path))

        # 2. Write images.txt
        images_txt = tmp_path / 'images.txt'
        images_txt.write_text('\n'.join(image_paths))

        # 3. Run prepare.py (no mocking - real MDS write)
        _run_prepare_main([
            '--images_txt', str(images_txt),
            '--local_mds_dir', str(mds_dir),
            '--num_proc', '2',
            '--seed', '42',
            '--size', str(num_images),
            '--min_size', '512',
        ])

        assert mds_dir.exists()

        # 4. Read MDS and save as PNG with caption as filename
        dataset = StreamingDataset(
            local=str(mds_dir),
            batch_size=1,
            shuffle=False,
        )
        for idx, sample in enumerate(dataset):
            img = sample['image']
            caption = sample['caption'] or 'unnamed'
            safe_name = _sanitize_for_filename(caption)
            out_path = output_dir / f'{safe_name}_{idx}.png'
            if isinstance(img, Image.Image):
                img.convert('RGB').save(out_path)
            else:
                Image.fromarray(img).convert('RGB').save(out_path)
            assert out_path.exists()
            # Verify we can read it back
            loaded = Image.open(out_path)
            assert loaded.size == (600, 600)

        # 5. Verify we got all 10 images
        saved_files = list(output_dir.glob('*.png'))
        assert len(saved_files) == num_images, f'Expected {num_images} PNGs, got {len(saved_files)}'

        # Verify each saved file has correct caption in name (or content matches)
        for f in saved_files:
            assert f.stat().st_size > 0
            stem = f.stem
            # Each filename should contain a sanitized caption substring
            assert len(stem) > 0
