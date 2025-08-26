"""
FastAPI application for TracePredicate web interface.

This module provides REST API endpoints for the TracePredicate system,
including device search, LDI scores, network visualization data, and analysis results.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.settings import get_settings
from ..database import get_db_session, get_database_statistics
from ..database.models import Device, LDIScore, AdverseEvent
from ..database.operations import DeviceOperations, LDIOperations

logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="TracePredicate API",
    description="Medical Device Regulatory Lineage Analysis System",
    version=settings.app.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API responses
class DeviceResponse(BaseModel):
    id: int
    k_number: str
    device_name: str
    applicant: str
    approval_date: Optional[datetime]
    device_code: Optional[str]
    product_code: Optional[str]
    decision: Optional[str]
    
    class Config:
        from_attributes = True


class LDIScoreResponse(BaseModel):
    id: int
    device_k_number: str
    ldi_score: float
    semantic_distance: Optional[float]
    parameter_difference: Optional[float]
    chain_length: Optional[int]
    calculation_date: datetime
    risk_percentile: Optional[float]
    
    class Config:
        from_attributes = True


class AdverseEventResponse(BaseModel):
    id: int
    mdr_report_key: str
    event_date: Optional[datetime]
    event_type: Optional[str]
    device_name: Optional[str]
    manufacturer: Optional[str]
    event_description: Optional[str]
    hfe_classification: Optional[str]
    
    class Config:
        from_attributes = True


class SystemStatusResponse(BaseModel):
    app_name: str
    version: str
    database_stats: Dict[str, int]
    uptime: str


class SearchResponse(BaseModel):
    results: List[DeviceResponse]
    total_count: int
    page: int
    page_size: int


# Dependency to get database session
def get_db():
    with get_db_session() as session:
        yield session


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "app": settings.app.app_name,
        "version": settings.app.app_version,
        "description": "Medical Device Regulatory Lineage Analysis System",
        "docs": "/docs"
    }


@app.get("/status", response_model=SystemStatusResponse, tags=["System"])
async def get_system_status():
    """Get system status and statistics."""
    try:
        stats = get_database_statistics()
        
        return SystemStatusResponse(
            app_name=settings.app.app_name,
            version=settings.app.app_version,
            database_stats=stats,
            uptime="Unknown"  # TODO: Track actual uptime
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@app.get("/devices", response_model=SearchResponse, tags=["Devices"])
async def search_devices(
    q: Optional[str] = Query(None, description="Search query for device name or K-number"),
    device_code: Optional[str] = Query(None, description="Filter by device code"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """Search for devices with optional filters."""
    try:
        query = db.query(Device)
        
        # Apply filters
        if device_code:
            query = query.filter(Device.device_code == device_code)
        
        if q:
            query = query.filter(
                Device.device_name.ilike(f"%{q}%") |
                Device.k_number.ilike(f"%{q}%") |
                Device.applicant.ilike(f"%{q}%")
            )
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        devices = query.offset(offset).limit(page_size).all()
        
        return SearchResponse(
            results=[DeviceResponse.from_orm(device) for device in devices],
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error searching devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to search devices")


@app.get("/devices/{k_number}", response_model=DeviceResponse, tags=["Devices"])
async def get_device(k_number: str, db: Session = Depends(get_db)):
    """Get a specific device by K-number."""
    try:
        device = DeviceOperations.get_device_by_k_number(db, k_number)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        return DeviceResponse.from_orm(device)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device {k_number}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get device")


@app.get("/devices/{k_number}/ldi", response_model=Optional[LDIScoreResponse], tags=["LDI"])
async def get_device_ldi(k_number: str, db: Session = Depends(get_db)):
    """Get the latest LDI score for a device."""
    try:
        ldi_score = LDIOperations.get_latest_ldi_score(db, k_number)
        if not ldi_score:
            return None
        
        # Create response with device K-number
        response_data = ldi_score.__dict__.copy()
        response_data['device_k_number'] = k_number
        
        return LDIScoreResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error getting LDI score for {k_number}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LDI score")


@app.get("/devices/{k_number}/predicates", response_model=List[DeviceResponse], tags=["Network"])
async def get_device_predicates(k_number: str, db: Session = Depends(get_db)):
    """Get predicate devices for a specific device."""
    try:
        # TODO: Use actual predicate operations
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error getting predicates for {k_number}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predicates")


@app.get("/devices/{k_number}/adverse-events", response_model=List[AdverseEventResponse], tags=["Safety"])
async def get_device_adverse_events(k_number: str, db: Session = Depends(get_db)):
    """Get adverse events for a specific device."""
    try:
        device = DeviceOperations.get_device_by_k_number(db, k_number)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        events = db.query(AdverseEvent).filter(AdverseEvent.device_id == device.id).all()
        
        return [AdverseEventResponse.from_orm(event) for event in events]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting adverse events for {k_number}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get adverse events")


@app.get("/ldi/scores", tags=["LDI"])
async def get_ldi_scores(
    device_code: Optional[str] = Query(None, description="Filter by device code"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum LDI score"),
    max_score: Optional[float] = Query(None, ge=0, le=1, description="Maximum LDI score"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """Get LDI scores with optional filters."""
    try:
        query = db.query(LDIScore, Device).join(Device, LDIScore.device_id == Device.id)
        
        # Apply filters
        if device_code:
            query = query.filter(Device.device_code == device_code)
        
        if min_score is not None:
            query = query.filter(LDIScore.ldi_score >= min_score)
            
        if max_score is not None:
            query = query.filter(LDIScore.ldi_score <= max_score)
        
        # Apply pagination
        offset = (page - 1) * page_size
        results = query.offset(offset).limit(page_size).all()
        
        # Format response
        scores = []
        for ldi_score, device in results:
            response_data = ldi_score.__dict__.copy()
            response_data['device_k_number'] = device.k_number
            scores.append(LDIScoreResponse(**response_data))
        
        return {
            "results": scores,
            "total_count": query.count(),
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Error getting LDI scores: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LDI scores")


@app.get("/ldi/statistics", tags=["LDI"])
async def get_ldi_statistics(
    device_code: Optional[str] = Query(None, description="Filter by device code"),
    db: Session = Depends(get_db)
):
    """Get LDI score statistics and percentiles."""
    try:
        percentiles = LDIOperations.calculate_ldi_percentiles(db, device_code)
        
        return {
            "device_code": device_code,
            "percentiles": percentiles
        }
        
    except Exception as e:
        logger.error(f"Error getting LDI statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LDI statistics")


@app.get("/network/graph", tags=["Network"])
async def get_network_graph(
    device_code: Optional[str] = Query(None, description="Filter by device code"),
    max_nodes: int = Query(100, ge=1, le=1000, description="Maximum nodes to include")
):
    """Get network graph data for visualization."""
    try:
        # TODO: Implement actual network graph generation
        # For now, return placeholder data
        return {
            "nodes": [],
            "edges": [],
            "device_code": device_code,
            "max_nodes": max_nodes
        }
        
    except Exception as e:
        logger.error(f"Error getting network graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to get network graph")


@app.get("/analysis/correlation", tags=["Analysis"])
async def get_correlation_analysis(device_code: Optional[str] = Query(None)):
    """Get correlation analysis between LDI scores and real-world outcomes."""
    try:
        # TODO: Implement actual correlation analysis
        return {
            "device_code": device_code,
            "correlation_coefficient": None,
            "p_value": None,
            "sample_size": 0,
            "message": "Correlation analysis not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get correlation analysis")


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    try:
        # Test database connection
        stats = get_database_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "total_devices": stats.get('devices', 0)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)