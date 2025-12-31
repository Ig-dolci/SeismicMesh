# benchmark the generation and quality of EAGE mesh
import time
import zipfile
import argparse
import numpy

# import meshplex  # Removed - requires license

# import meshio
import pygalmesh
import pygmsh
import gmsh

from SeismicMesh import (
    get_sizing_function_from_segy,
    Cube,
    generate_mesh,
    sliver_removal,
)


from helpers import print_stats_3d


# Alternative quality computation without meshplex
def compute_tetrahedron_quality(points, cells):
    """
    Compute mesh quality metrics for tetrahedral meshes without meshplex.
    
    Returns:
        angles: minimum sine of dihedral angles
        quality: radius ratio quality metric
    """
    def tetrahedron_volume(p0, p1, p2, p3):
        """Compute volume of tetrahedron."""
        return numpy.abs(numpy.dot(p1 - p0, numpy.cross(p2 - p0, p3 - p0))) / 6.0
    
    def edge_lengths(p0, p1, p2, p3):
        """Compute all 6 edge lengths of tetrahedron."""
        edges = [
            numpy.linalg.norm(p1 - p0),
            numpy.linalg.norm(p2 - p0),
            numpy.linalg.norm(p3 - p0),
            numpy.linalg.norm(p2 - p1),
            numpy.linalg.norm(p3 - p1),
            numpy.linalg.norm(p3 - p2),
        ]
        return numpy.array(edges)
    
    def circumradius(p0, p1, p2, p3):
        """Compute circumradius of tetrahedron."""
        vol = tetrahedron_volume(p0, p1, p2, p3)
        if vol < 1e-15:
            return 1e10
        
        # Edge vectors
        a = numpy.linalg.norm(p1 - p0)
        b = numpy.linalg.norm(p2 - p1)
        c = numpy.linalg.norm(p0 - p2)
        d = numpy.linalg.norm(p3 - p0)
        e = numpy.linalg.norm(p3 - p1)
        f = numpy.linalg.norm(p3 - p2)
        
        # Use formula for circumradius
        denom = 24.0 * vol
        if abs(denom) < 1e-15:
            return 1e10
        
        term1 = a * f
        term2 = b * d
        term3 = c * e
        
        R = numpy.sqrt(term1**2 + term2**2 + term3**2) / denom
        return R
    
    def inradius(p0, p1, p2, p3):
        """Compute inradius of tetrahedron."""
        vol = tetrahedron_volume(p0, p1, p2, p3)
        
        # Compute face areas
        face_area = lambda pa, pb, pc: 0.5 * numpy.linalg.norm(numpy.cross(pb - pa, pc - pa))
        
        A1 = face_area(p0, p1, p2)
        A2 = face_area(p0, p1, p3)
        A3 = face_area(p0, p2, p3)
        A4 = face_area(p1, p2, p3)
        
        surface_area = A1 + A2 + A3 + A4
        
        if surface_area < 1e-15:
            return 0.0
        
        return 3.0 * vol / surface_area
    
    # Compute quality for all tetrahedra
    n_cells = len(cells)
    radius_ratios = numpy.zeros(n_cells)
    min_angles = numpy.zeros(n_cells)
    
    for i, cell in enumerate(cells):
        p0, p1, p2, p3 = points[cell[0]], points[cell[1]], points[cell[2]], points[cell[3]]
        
        # Radius ratio quality (inradius / circumradius)
        R = circumradius(p0, p1, p2, p3)
        r = inradius(p0, p1, p2, p3)
        
        if R > 0:
            radius_ratios[i] = r / R
        else:
            radius_ratios[i] = 0.0
        
        # Approximate dihedral angle quality using edge length ratios
        edges = edge_lengths(p0, p1, p2, p3)
        vol = tetrahedron_volume(p0, p1, p2, p3)
        
        # Normalized quality metric (0 = degenerate, 1 = regular)
        edge_ratio = numpy.min(edges) / numpy.max(edges) if numpy.max(edges) > 0 else 0.0
        min_angles[i] = edge_ratio  # Approximation
    
    return min_angles, radius_ratios

# Bounding box describing domain extents (corner coordinates)
bbox = (-4200.0, 0.0, 0.0, 13520.0, 0.0, 13520.0)


def _build_sizing(HMIN=150.0, FREQ=2):

    # This file is in a big Endian binary format, so we must tell the program the shape of the velocity model.
    path = "Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/"
    # Extract binary file Saltf@@ from SALTF.ZIP
    zipfile.ZipFile(path + "SALTF.ZIP", "r").extract("Saltf@@", path=path)

    fname = path + "Saltf@@"

    # Dimensions of model (number of grid points in z, x, and y)
    nx, ny, nz = 676, 676, 210

    ef = get_sizing_function_from_segy(
        fname,
        bbox,
        hmin=HMIN,
        dt=0.001,
        freq=FREQ,
        wl=5,
        grade=0.15,
        hmax=5e3,
        nz=nz,
        nx=nx,
        ny=ny,
        byte_order="big",
        axes_order=[2, 0, 1],  # default order z, x, y -> order for EAGE x, y, z
        axes_order_sort="F",  # binary is packed in a FORTRAN-style
    )

    return ef


# for pytest-benchmark
# ef = _build_sizing()


def test_seismic_mesh(benchmark):
    angles, quality, elapsed, num_vertices, num_cells = benchmark.pedantic(
        run_SeismicMesh, args=(ef), iterations=1, rounds=5, warmup_rounds=0
    )
    assert numpy.amin(angles / numpy.pi * 180) > 10.0


def test_gmsh(benchmark):
    angles, quality, elapsed, num_vertices, num_cells = benchmark.pedantic(
        run_gmsh, args=(ef), iterations=1, rounds=5, warmup_rounds=0
    )
    assert numpy.amin(angles / numpy.pi * 180) > 10.0


def test_cgal(benchmark):
    angles, quality, elapsed, num_vertices, num_cells = benchmark.pedantic(
        run_cgal, args=(ef), iterations=1, rounds=5, warmup_rounds=0
    )
    assert numpy.amin(angles / numpy.pi * 180) > 10.0


def run_cgal(ef, HMIN=75.0):

    print("generating a mesh with cgal...")
    t1 = time.time()
    mesh = pygalmesh.generate_mesh(
        pygalmesh.Cuboid([-4200.0, 0.0, 0.0], [0.0, 13520.0, 13520.0]),
        facet_angle=30,
        cell_radius_edge_ratio=2.0,
        cell_size=lambda x: ef.eval(x) / 1.1,
        edge_size=HMIN,
    )
    elapsed = time.time() - t1

    # mesh.write("cgal_EAGE.vtk")

    # Use custom quality computation instead of meshplex
    angles, quality = compute_tetrahedron_quality(mesh.points, mesh.cells[1][1])

    num_cells = len(mesh.cells[1][1])
    num_vertices = len(mesh.points)

    return angles, quality, elapsed, num_vertices, num_cells


def run_gmsh(ef, HMIN=75.0):
    with pygmsh.geo.Geometry() as geom:

        geom.add_box(-4200.0, 0.0, 0.0, 13520.0, 0.0, 13520.0, HMIN)

        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z: (ef.eval([x, y, z])) / 1.1
        )
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        print("generating a mesh with gmsh...")
        t1 = time.time()
        mesh = geom.generate_mesh()
        elapsed = time.time() - t1

        points = mesh.points
        cells = mesh.cells[2].data

        num_cells = len(cells)
        num_vertices = len(points)

        # Use custom quality computation instead of meshplex
        angles, quality = compute_tetrahedron_quality(points, cells)

        # mesh.write("gmsh_EAGE.vtk")

        return angles, quality, elapsed, num_vertices, num_cells


def run_SeismicMesh(ef, HMIN=75.0):

    cube = Cube(bbox)

    t1 = time.time()
    points, cells = generate_mesh(
        domain=cube,
        edge_length=ef,
        max_iter=25,
    )

    points, cells = sliver_removal(
        points=points,
        domain=cube,
        edge_length=ef,
    )
    elapsed = time.time() - t1

    # meshio.write_points_cells(
    #    "sm_EAGE.vtk",
    #    points,
    #    [("tetra", cells)],
    #    file_format="vtk",
    # )

    # Use custom quality computation instead of meshplex
    angles, quality = compute_tetrahedron_quality(points, cells)

    num_cells = len(cells)
    num_vertices = len(points)

    return angles, quality, elapsed, num_vertices, num_cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default=None,
        required=False,
        help="Run benchmark with method=('cgal','sm', 'gmsh')",
    )

    args = parser.parse_args()

    ef = _build_sizing()
    if args.method == "cgal":
        a1, q1, t1, nv, nc = run_cgal(ef)
        print_stats_3d(a1, q1, "cgal", t1, nv, nc)
    elif args.method == "sm":
        a1, q1, t1, nv, nc = run_SeismicMesh(ef)
        print_stats_3d(a1, q1, "SeismicMesh", t1, nv, nc)
    elif args.method == "gmsh":
        a1, q1, t1, nv, nc = run_gmsh(ef)
        print_stats_3d(a1, q1, "gmsh", t1, nv, nc)
    else:
        # a1, q1, t1, nv1, nc1 = run_cgal(ef)
        a2, q2, t2, nv2, nc2 = run_SeismicMesh(ef)
        a3, q3, t3, nv3, nc3 = run_gmsh(ef)
        # print_stats_3d(a1, q1, "CGAL", t1, nv1, nc1)
        print_stats_3d(a2, q2, "SeismicMesh", t2, nv2, nc2)
        print_stats_3d(a3, q3, "gmsh", t3, nv3, nc3)
