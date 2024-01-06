import numpy as np

from utils import *
from ImLite import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, texture = None):
        """Create a new material with the given parameterself.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d
        self.texture = texture 


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

# Value to represent absence of an intersection
no_hit = Hit(np.inf)
img = Image(path = "./ref/squares.png").GetFloatCopy().pixels
# print(img)

class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radiuself.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.origin-self.center, ray.direction)
        c = np.dot(ray.origin-self.center, ray.origin-self.center) - self.radius ** 2
        disc = b ** 2 - 4 * a * c

        if disc >= 0: # quadratic func has sol 
            t1 = (-b + np.sqrt(disc)) / (2*a)
            t2 = (-b - np.sqrt(disc)) / (2*a)

            t1_valid = t1 >= ray.start and t1 <= ray.end
            t2_valid = t2 >= ray.start and t2 <= ray.end
            if t1_valid and t2_valid:
                t = min(t1, t2)
            elif t1_valid:
                t = t1 
            elif t2_valid:
                t = t2 
            else:
                return no_hit 

            point = ray.origin + t * ray.direction
            normal = (point - self.center) / self.radius
            
            # if self.material.texture is not None:
            #     # img = Image(path = self.material.texture).GetFloatCopy().pixels
            #     u = int(normal[0]/2 + 0.5)
            #     v = int(normal[1]/2 + 0.5)

            #     color = img[u, v]
            #     self.material.k_d = from_srgb8(color)
                
            return Hit(t, point, normal, self.material)
        else:
            return no_hit

class Box:
    def __init__(self, min_corner, max_corner, material):
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.material = material

    def intersect(self, ray):
        tmin = np.minimum((self.min_corner - ray.origin) / ray.direction, (self.max_corner - ray.origin) / ray.direction)
        tmax = np.maximum((self.min_corner - ray.origin) / ray.direction, (self.max_corner - ray.origin) / ray.direction)

        t_enter = np.max(tmin)
        t_exit = np.min(tmax)

        if t_enter > t_exit or t_exit < ray.start:
            return no_hit

        enter_point = ray.origin + t_enter * ray.direction
        exit_point = ray.origin + t_exit * ray.direction

        hit_point = exit_point 
        if t_enter > ray.start:
            hit_point = enter_point
        normal = self.calculate_normal(hit_point)

        return Hit(t_enter, hit_point, normal, self.material)

    def calculate_normal(self, point):
        """Calculate the normal of the cube at a given point."""
        epsilon = 1e-6
        x, y, z = point

        if np.abs(x - self.min_corner[0]) < epsilon:
            return vec([-1, 0, 0])
        elif np.abs(x - self.max_corner[0]) < epsilon:
            return vec([1, 0, 0])
        elif np.abs(y - self.min_corner[1]) < epsilon:
            return vec([0, -1, 0])
        elif np.abs(y - self.max_corner[1]) < epsilon:
            return vec([0, 1, 0])
        elif np.abs(z - self.min_corner[2]) < epsilon:
            return vec([0, 0, -1])
        elif np.abs(z - self.max_corner[2]) < epsilon:
            return vec([0, 0, 1])
        else:
            return vec([0, 0, 0])  

class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given verticeself.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it existself.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        edge1 = self.vs[1] - self.vs[0]
        edge2 = self.vs[2] - self.vs[0]

        # get the cross product of an edge and the ray
        h = np.cross(ray.direction, edge2)
       
        # a-- to see where the intersection is
        a = np.dot(edge1, h)

        if a > -1e-6 and a < 1e-6:
            return no_hit  

        s = ray.origin - self.vs[0]
        u = 1 / a * np.dot(s, h)

        if u < 0 or u > 1:
            return no_hit

        v = 1 / a * np.dot(ray.direction, np.cross(s, edge1))

        if v < 0 or u + v > 1:
            return no_hit

        t = 1 / a * np.dot(edge2, np.cross(s, edge1))

        if t >= ray.start and t <= ray.end:
            hit_point = ray.origin + t * ray.direction
            return Hit(t, hit_point, np.cross(edge1, edge2), self.material)

        return no_hit


class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameterself.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.target = target
        self.aspect = aspect
        self.up = up 
        self.vfov = vfov
        
        # self.f = np.linalg.norm(target-eye)
        self.f = 1
        self.height = self.f * np.tan(vfov/2 * np.pi/180) * 2
        self.width = aspect * self.height

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
                      (note: since we initially released this code with specs saying 0,0 was at the bottom left, we will
                       accept either convention for this assignment)
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        img_point[1]=1-img_point[1]
        
        w = normalize(self.eye - self.target)
        u = normalize(np.cross(self.up, w))
        v = normalize(np.cross(w, u))

        self.f = 1 / np.tan(self.vfov/2 * np.pi/180)
        
        d = np.linalg.norm(self.target-self.eye)  
        pt_x = 2*(img_point[0]) - 1
        pt_y = 2*(img_point[1]) - 1
        img_pt = np.array([pt_x, pt_y, 1])      
        direction = -self.f*w + img_pt[0] * u * self.aspect + img_pt[1] * v        
        
        return Ray(self.eye, direction)
        

def normalize(v):
        return v / np.linalg.norm(v)
    
class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        r = np.linalg.norm(self.position - hit.point)
        irradiance = (max(0, np.dot(hit.normal, normalize(self.position - hit.point))))/r**2 * self.intensity 

        # specular 
        v = -normalize(ray.direction)
        h = (v + normalize(self.position - hit.point))/np.linalg.norm(v+normalize(self.position - hit.point))
        p = hit.material.p
        specular = hit.material.k_s * (np.dot(normalize(hit.normal),h) ** p)

        ray_pos = hit.point + 1e-9 * (self.position - hit.point)

        shadRay = Ray(ray_pos, self.position - hit.point)
        if scene.intersect(shadRay) == no_hit:
            return (hit.material.k_d + specular) * irradiance

        return vec([0,0,0])
        # return hit.material.k_a

        # return (hit.material.k_d + specular) * irradiance
        # return hit.material.k_d * irradiance
        

class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        return hit.material.k_a * self.intensity 
        # return vec([0,0,0])


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objectself.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        small_hit = no_hit 
        
        for shape in self.surfs:
            intersect = shape.intersect(ray)
            if intersect.t < small_hit.t:
                small_hit = intersect 
                
        return small_hit

class CSG:
    
    def __init__(self, shape1, shape2, operation):
        self.shape1 = shape1 
        self.shape2 = shape2
        self.operation = operation

    def intersect(self, ray):
        hit1 = self.shape1.intersect(ray)
        hit2 = self.shape2.intersect(ray)

        if hit1 == no_hit or hit2 == no_hit:
            return no_hit 

        if self.operation == "intersect":
            if hit1.t > hit2.t:
                return hit1
            else:
                return hit2

        if self.operation == "union":
            if hit1.t < hit2.t:
                return hit1
            else:
                return hit2

        if self.operation == "subtract":
            new_ray = Ray(hit2.point + 1e-9, ray.direction)
            hit3 = self.shape1.intersect(new_ray)
            # hit3.material = Material(vec([0.0, .8, 0.0]), k_s=0.3, p=90, k_m=0.3)
            return hit3

        return no_hit 

MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # ambient = vec([0,0,0])
    # # ambient = hit.material.k_d
    # for light in lights:
    #     ambient += light.illuminate(ray, hit, scene)
    # return ambient
    
    if hit == no_hit:
        return scene.bg_color
    if depth == MAX_DEPTH + 1:
        ambient = vec([0,0,0])
        # ambient = hit.material.k_d
        for light in lights:
            ambient += light.illuminate(ray, hit, scene)
        return ambient 

    ambient = vec([0,0,0])
    # ambient = hit.material.k_d
    for light in lights:
        ambient += light.illuminate(ray, hit, scene)

    v = -normalize(ray.direction)
    r = 2*(np.dot(np.dot(hit.normal, v),hit.normal)) - v
    past_hit = hit
    ray = Ray(hit.point + 1e-9*r, r)
    hit = scene.intersect(ray)
    return ambient + past_hit.material.k_m * shade(ray, hit, scene, lights, depth+1)


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function
    output_image = np.zeros((ny,nx,3), np.float32)
    # matrix = np.array([[nx, 0, -nx/2],
    #                     [0 ,-ny,ny/2],
    #                    [0,0,1]])
    for i in range(ny):
        for j in range(nx):

            u = (j + 0.5)/nx 
            v = (i + 0.5)/ny
            # u = j/nx
            # v = i/ny
            
            ray = camera.generate_ray(vec([u,v]))
            intersection = scene.intersect(ray); # this will return a Hit object

            if intersection == no_hit:  
                output_image[i, j] = scene.bg_color 
            else:
                output_image[i, j] = shade(ray, intersection, scene, lights)
    
    return output_image
    # return np.zeros((ny,nx,3), np.float32)
        

# def c2():
#     importlib.reload(ray)
#     tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
#     blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
#     gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

#     sphere = ray.Sphere(vec([0, 0, 0]), 0.6, tan)
#     cube = ray.Cube(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), blue)
    
#     scene = ray.Scene([
#         ray.Sphere(vec([0, 0, 0]), 0.6, tan),
#         ray.Cube(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), blue),
#         ray.Sphere(vec([0, -40, 0]), 39.5, gray)
#     ])

#     lights = [
#         ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#         ray.AmbientLight(0.1),
#     ]

#     camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9)
#     return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
