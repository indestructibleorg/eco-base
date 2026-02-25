#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: deploy
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Universal Deployment Script
Deploys Machine Native Ops to any environment
"""
import asyncio
import argparse
import sys
import logging
from pathlib import Path
# Add ns-root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'ns-root'))
from adk.plugins.deployment import (
    UniversalDeploymentManager,
    EnvironmentDetector,
    ProviderAdapterFactory
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description='Universal Deployment Script for Machine Native Ops'
    )
    parser.add_argument(
        '--provider',
        type=str,
        help='Provider to deploy to (aws, gcp, azure, kubernetes, docker-compose)'
    )
    parser.add_argument(
        '--environment',
        type=str,
        default='production',
        help='Environment to deploy to (default: production)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Custom configuration file path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - do not actually deploy'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect environment'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Enable automatic rollback on failure'
    )
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='List available providers and exit'
    )
    parser.add_argument(
        '--detect-env',
        action='store_true',
        help='Detect current environment and exit'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    args = parser.parse_args()
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    # List providers
    if args.list_providers:
        providers = ProviderAdapterFactory.list_providers()
        print("Available providers:")
        for provider in sorted(providers):
            print(f"  - {provider}")
        return 0
    # Detect environment
    if args.detect_env:
        detector = EnvironmentDetector()
        env_info = detector.detect()
        print(f"Environment: {env_info.type}")
        print(f"Provider: {env_info.provider}")
        if env_info.region:
            print(f"Region: {env_info.region}")
        if env_info.version:
            print(f"Version: {env_info.version}")
        return 0
    # Validate arguments
    if not args.provider and not args.auto_detect:
        print("Error: Either --provider or --auto-detect must be specified")
        parser.print_help()
        return 1
    # Create deployment manager
    try:
        logger.info(f"Initializing deployment manager...")
        logger.info(f"Provider: {args.provider or 'auto-detect'}")
        logger.info(f"Environment: {args.environment}")
        manager = UniversalDeploymentManager(
            provider=args.provider,
            environment=args.environment,
            config_file=args.config,
            auto_detect=args.auto_detect,
            dry_run=args.dry_run,
            auto_rollback=args.rollback
        )
        # Deploy
        logger.info(f"Starting deployment...")
        result = await manager.deploy()
        # Print results
        print("\n" + "="*50)
        print("Deployment Results")
        print("="*50)
        print(f"Success: {result.success}")
        print(f"Provider: {result.provider}")
        print(f"Environment: {result.environment}")
        print(f"Duration: {result.duration:.2f}s")
        if result.resources:
            print(f"\nResources Created:")
            for resource_type, resources in result.resources.items():
                print(f"  {resource_type}: {len(resources)} items")
        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        print("="*50 + "\n")
        return 0 if result.success else 1
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        return 1
if __name__ == '__main__':
    sys.exit(asyncio.run(main()))