"""
Command Line Interface for TracePredicate.

This module provides a comprehensive CLI for the TracePredicate system,
including data collection, analysis, and database management commands.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config.settings import get_settings
from .database import init_database, get_database_statistics, get_db_session
from .database.operations import DeviceOperations, LDIOperations
from .data_collection.fda_510k_scraper import FDA510KScraper
from .data_collection.maude_interface import MAUDEInterface
from .data_collection.fda_recall_scraper import FDARecallScraper

# Initialize CLI app
app = typer.Typer(
    name="trace-predicate",
    help="TracePredicate: Medical Device Regulatory Lineage Analysis System"
)

# Initialize console for rich output
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.command()
def init_db(
    drop_existing: bool = typer.Option(
        False, 
        "--drop", 
        help="Drop existing tables before creating new ones"
    )
):
    """Initialize the TracePredicate database."""
    console.print("🔧 Initializing TracePredicate database...", style="blue")
    
    try:
        init_database(drop_existing=drop_existing)
        
        # Show database statistics
        stats = get_database_statistics()
        
        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Records", justify="right", style="green")
        
        for table_name, count in stats.items():
            table.add_row(table_name.replace('_', ' ').title(), str(count))
        
        console.print(table)
        console.print("✅ Database initialized successfully!", style="green")
        
    except Exception as e:
        console.print(f"❌ Database initialization failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def collect(
    device_code: str = typer.Option("KWA", "--device-code", "-d", help="FDA device code to collect"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of devices to collect"),
    include_adverse_events: bool = typer.Option(True, "--adverse-events", help="Also collect MAUDE adverse events"),
    include_recalls: bool = typer.Option(True, "--recalls", help="Also collect FDA recalls")
):
    """Collect FDA 510(k) devices and related data."""
    settings = get_settings()
    
    console.print(f"📥 Collecting data for device code: {device_code}", style="blue")
    if limit:
        console.print(f"   Limited to {limit} devices")
    
    try:
        # Collect 510(k) devices
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Task 1: Collect 510(k) devices
            task1 = progress.add_task("Collecting 510(k) devices...", total=None)
            
            scraper = FDA510KScraper()
            devices = asyncio.run(scraper.collect_device_family(device_code, limit))
            
            progress.update(task1, description=f"✅ Collected {len(devices)} devices")
            
            # Store devices in database
            task2 = progress.add_task("Storing devices in database...", total=len(devices))
            
            stored_count = 0
            with get_db_session() as session:
                for i, device in enumerate(devices):
                    try:
                        DeviceOperations.create_device(
                            session,
                            k_number=device.k_number,
                            device_name=device.device_name,
                            applicant=device.applicant,
                            approval_date=device.approval_date,
                            device_code=device.device_code,
                            product_code=device.product_code,
                            decision=device.decision,
                            summary_url=device.summary_url,
                            pdf_content=device.pdf_content,
                            technical_parameters=device.technical_parameters
                        )
                        stored_count += 1
                    except Exception as e:
                        logger.warning(f"Error storing device {device.k_number}: {e}")
                    
                    progress.update(task2, advance=1)
            
            progress.update(task2, description=f"✅ Stored {stored_count} devices")
            
            # Collect adverse events if requested
            if include_adverse_events:
                task3 = progress.add_task("Collecting adverse events...", total=None)
                
                maude = MAUDEInterface()
                events_data = maude.search_events_by_product_code(device_code, limit=1000)
                
                progress.update(task3, description=f"✅ Collected {len(events_data)} adverse events")
            
            # Collect recalls if requested
            if include_recalls:
                task4 = progress.add_task("Collecting recalls...", total=None)
                
                recall_scraper = FDARecallScraper()
                recalls_data = recall_scraper.search_recalls_by_product_code(device_code, limit=500)
                
                progress.update(task4, description=f"✅ Collected {len(recalls_data)} recalls")
        
        console.print(f"✅ Data collection completed successfully!", style="green")
        console.print(f"   Devices: {len(devices)}")
        if include_adverse_events:
            console.print(f"   Adverse Events: {len(events_data)}")
        if include_recalls:
            console.print(f"   Recalls: {len(recalls_data)}")
            
    except Exception as e:
        console.print(f"❌ Data collection failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def calculate_ldi(
    device_code: Optional[str] = typer.Option(None, "--device-code", "-d", help="Calculate LDI for specific device code"),
    k_number: Optional[str] = typer.Option(None, "--k-number", "-k", help="Calculate LDI for specific K-number"),
    force_recalculate: bool = typer.Option(False, "--force", help="Recalculate even if scores exist"),
    semantic_weight: float = typer.Option(0.4, "--semantic-weight", help="Weight for semantic distance"),
    parameter_weight: float = typer.Option(0.4, "--parameter-weight", help="Weight for parameter difference"),
    chain_weight: float = typer.Option(0.2, "--chain-weight", help="Weight for chain length")
):
    """Calculate Lineage Drift Index (LDI) scores for devices."""
    
    # Validate weights
    total_weight = semantic_weight + parameter_weight + chain_weight
    if abs(total_weight - 1.0) > 0.001:
        console.print(f"❌ Weights must sum to 1.0, got {total_weight:.3f}", style="red")
        raise typer.Exit(1)
    
    console.print("🧮 Calculating LDI scores...", style="blue")
    
    try:
        with get_db_session() as session:
            # Get devices to process
            if k_number:
                devices = [DeviceOperations.get_device_by_k_number(session, k_number)]
                devices = [d for d in devices if d is not None]
            elif device_code:
                devices = DeviceOperations.get_devices_by_code(session, device_code)
            else:
                console.print("❌ Must specify either --device-code or --k-number", style="red")
                raise typer.Exit(1)
            
            if not devices:
                console.print("❌ No devices found to process", style="red")
                raise typer.Exit(1)
            
            console.print(f"Processing {len(devices)} devices...")
            
            calculated_count = 0
            skipped_count = 0
            
            with Progress() as progress:
                task = progress.add_task("Calculating LDI scores...", total=len(devices))
                
                for device in devices:
                    # Check if LDI already exists
                    if not force_recalculate:
                        existing_score = LDIOperations.get_latest_ldi_score(session, device.k_number)
                        if existing_score:
                            skipped_count += 1
                            progress.advance(task)
                            continue
                    
                    # TODO: Implement actual LDI calculation
                    # For now, create placeholder score
                    dummy_score = 0.5  # This would be calculated from actual algorithm
                    
                    LDIOperations.create_ldi_score(
                        session,
                        device_k_number=device.k_number,
                        ldi_score=dummy_score,
                        semantic_distance=0.3,
                        parameter_difference=0.4,
                        chain_length=3,
                        semantic_weight=semantic_weight,
                        parameter_weight=parameter_weight,
                        chain_weight=chain_weight,
                        calculation_version="0.1.0"
                    )
                    
                    calculated_count += 1
                    progress.advance(task)
        
        console.print(f"✅ LDI calculation completed!", style="green")
        console.print(f"   Calculated: {calculated_count}")
        console.print(f"   Skipped: {skipped_count}")
        
    except Exception as e:
        console.print(f"❌ LDI calculation failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def analyze(
    analysis_type: str = typer.Argument(..., help="Type of analysis: 'network', 'correlation', 'validation'"),
    device_code: Optional[str] = typer.Option(None, "--device-code", "-d", help="Analyze specific device code"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results")
):
    """Run various analyses on the collected data."""
    console.print(f"📊 Running {analysis_type} analysis...", style="blue")
    
    try:
        if analysis_type == "network":
            console.print("🕸️  Network analysis not yet implemented", style="yellow")
        elif analysis_type == "correlation":
            console.print("📈 Correlation analysis not yet implemented", style="yellow")
        elif analysis_type == "validation":
            console.print("✔️  Validation analysis not yet implemented", style="yellow")
        else:
            console.print(f"❌ Unknown analysis type: {analysis_type}", style="red")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("localhost", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development")
):
    """Start the web interface server."""
    console.print(f"🌐 Starting web server on {host}:{port}...", style="blue")
    
    try:
        import uvicorn
        uvicorn.run(
            "trace_predicate.api.main:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        console.print("❌ uvicorn not installed. Install with: pip install uvicorn", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start server: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def status():
    """Show system status and statistics."""
    console.print("📊 TracePredicate System Status", style="blue")
    
    try:
        # Database statistics
        stats = get_database_statistics()
        
        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Records", justify="right", style="green")
        
        for table_name, count in stats.items():
            table.add_row(table_name.replace('_', ' ').title(), str(count))
        
        console.print(table)
        
        # Settings info
        settings = get_settings()
        console.print(f"\n🔧 Configuration:")
        console.print(f"  Database: {settings.database.host}:{settings.database.port}/{settings.database.database}")
        console.print(f"  Data Directory: {settings.app.data_directory}")
        console.print(f"  Debug Mode: {settings.app.debug}")
        
        # Validation warnings
        warnings = settings.validate_settings()
        if warnings:
            console.print(f"\n⚠️  Configuration Warnings:", style="yellow")
            for warning in warnings:
                console.print(f"    • {warning}", style="yellow")
        else:
            console.print(f"\n✅ Configuration validated successfully!", style="green")
        
    except Exception as e:
        console.print(f"❌ Error getting system status: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def export(
    table: str = typer.Argument(..., help="Table to export: devices, adverse_events, recalls, ldi_scores"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output CSV file"),
    device_code: Optional[str] = typer.Option(None, "--device-code", "-d", help="Filter by device code"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of records")
):
    """Export data to CSV files."""
    console.print(f"💾 Exporting {table} data...", style="blue")
    
    try:
        # TODO: Implement actual export functionality
        console.print(f"📁 Export functionality not yet implemented", style="yellow")
        console.print(f"   Would export {table} to {output_file}")
        if device_code:
            console.print(f"   Filtered by device code: {device_code}")
        if limit:
            console.print(f"   Limited to {limit} records")
            
    except Exception as e:
        console.print(f"❌ Export failed: {e}", style="red")
        raise typer.Exit(1)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all output except errors")
):
    """
    TracePredicate: Medical Device Regulatory Lineage Analysis System
    
    A comprehensive platform for analyzing FDA 510(k) medical device regulatory
    lineages and calculating Lineage Drift Index (LDI) for risk assessment.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)


if __name__ == "__main__":
    app()