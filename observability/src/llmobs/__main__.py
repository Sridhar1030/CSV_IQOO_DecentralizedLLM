"""`llmobs-server` entry point."""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llmobs-server",
        description="Run the llmobs OTLP proxy in front of an OTel Collector.",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--upstream", default=None, help="The real OTel Collector to forward to."
    )
    parser.add_argument("--service-name", default=None)
    parser.add_argument("--console-export", action="store_true", default=None)
    parser.add_argument("--no-otlp", action="store_true", help="Disable OTLP export.")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(
        server_host=args.host,
        server_port=args.port,
        otlp_endpoint=args.upstream,
        service_name=args.service_name,
        console_export=args.console_export,
        otlp_enabled=False if args.no_otlp else None,
    )

    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(
            otlp_endpoint=config.otlp_endpoint,
            service_name=config.service_name,
            console_export=config.console_export,
            otlp_enabled=config.otlp_enabled,
        ),
        host=config.server_host,
        port=config.server_port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
