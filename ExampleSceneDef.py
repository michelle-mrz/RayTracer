import ray
from ImLite import *
from utils import *
import importlib

class ExampleSceneDef(object):
    def __init__(self, camera, scene, lights):
        self.camera = camera;
        self.scene = scene;
        self.lights = lights;

    def render(self, output_path=None, output_shape=None, gamma_correct=True, srgb_whitepoint=None):
        importlib.reload(ray)
        if(output_shape is None):
            output_shape=[128,128];
        if(srgb_whitepoint is None):
            srgb_whitepoint = 1.0;
        pix = ray.render_image(self.camera, self.scene, self.lights, output_shape[1], output_shape[0]);
        im = None;
        if(gamma_correct):
            cam_img_ui8 = to_srgb8(pix / srgb_whitepoint)
            im = Image(pixels=cam_img_ui8);
        else:
            im = im = Image(pixels=pix);
        if(output_path is None):
            return im;
        else:
            im.writeToFile(output_path);


def TwoSpheresExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    scene = ray.Scene([
        ray.Sphere(vec([0, 0, 0]), 0.5, tan),
        ray.Sphere(vec([0, -40, 0]), 39.5, gray),
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]
    camera = ray.Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);


def ThreeSpheresExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

    scene = ray.Scene([
        ray.Sphere(vec([-0.7, 0, 0]), 0.5, tan),
        ray.Sphere(vec([0.7, 0, 0]), 0.5, blue),
        ray.Sphere(vec([0, -40, 0]), 39.5, gray),
    ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def c2():
    importlib.reload(ray)
    tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    blue1 = ray.Material(vec([0.2, 0.2, 0.5]))
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
    # white = ray.Material(vec([1.0, 0.88, 0.65]), k_m=0.5)
    white = ray.Material(vec([1.3, 0.88, 0.65]), k_s=0.3, p=90, k_m=0.8)
    green = ray.Material(vec([0.0, .8, 0.0]), k_s=0.3, p=90, k_m=0.3)

    # sphere = ray.Sphere(vec([0, 0, 0]), 0.66, white)
    sphere = ray.Sphere(vec([0, 0, 0]), 0.70, white)
    # sphere = ray.Sphere(vec([0, 0, 0]), 0.78, white)
    cube = ray.Box(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), white)
    texture = ray.Material(vec([0.2, 0.2, 0.2]), texture="./ref/squares.png")
    # csg_intersection = CSG(sphere, cube, "intersect")

    second_cube = ray.Box(vec([2, 0.5, 0.5]), vec([1, -.5, -.5]), blue)
    second_sphere = ray.Sphere(vec([1.5, 0, 0]), 0.66, blue)
    shape1 = ray.CSG(second_sphere, second_cube, "subtract")
    

    five_dice_1 = ray.Sphere(vec([0.15, 0.15, 0.5]), 0.075, blue)
    five_dice_2 = ray.Sphere(vec([0, 0, 0.5]), 0.075, blue)
    five_dice_3 = ray.Sphere(vec([-0.15, -0.15, 0.5]), 0.075, blue)
    five_dice_4 = ray.Sphere(vec([0.15, -0.15, 0.5]), 0.075, blue)
    five_dice_5 = ray.Sphere(vec([-0.15, 0.15, 0.5]), 0.075, blue)

    four_dice_1 = ray.Sphere(vec([0.5, -0.15, -0.15]), 0.075, blue)
    four_dice_2 = ray.Sphere(vec([0.5, 0, 0]), 0.075, blue)
    four_dice_3 = ray.Sphere(vec([0.5, 0.15, 0.15]), 0.075, blue)
    four_dice_4 = ray.Sphere(vec([0.5, -0.15, 0.15]), 0.075, blue)
    four_dice_5 = ray.Sphere(vec([0.5, 0.15, -0.15]), 0.075, blue)

    three_dice_1 = ray.Sphere(vec([0, 0.5, 0]), 0.075, blue)
    three_dice_2 = ray.Sphere(vec([0.15, 0.5, -0.15]), 0.075, blue)
    three_dice_3 = ray.Sphere(vec([-0.15, 0.5, 0.15]), 0.075, blue)
    three_dice_4 = ray.Sphere(vec([-0.15, 0.5, -0.15]), 0.075, blue)
    three_dice_5 = ray.Sphere(vec([0.15, 0.5, 0.15]), 0.075, blue)

    floor_sphere_1 = ray.Sphere(vec([1.5, -0.5, -0.5]), 0.2, texture)

    another_cube = ray.Box(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), blue)

    dice = ray.CSG(sphere, cube, "intersect")

    rectangle1 = ray.Box(vec([1,0,-1]), vec([2,1,-0.2]), white)
    
    scene = ray.Scene([
        # floor_sphere_1,
        dice,
        # rectangle1, 
        # second_cube,
        # second_sphere,
        # shape1,
        ray.Sphere(vec([0, -40, 0]), 39.5, green),
        # ray.CSG(ray.Sphere(vec([0, -40, 0]), 39.5, green), floor_sphere_1, "subtract"),
        ray.CSG(dice, five_dice_1, "union"),
        ray.CSG(dice, five_dice_2, "union"),
        ray.CSG(dice, five_dice_3, "union"),
        ray.CSG(dice, five_dice_4, "union"),
        ray.CSG(dice, five_dice_5, "union"),
        ray.CSG(dice, four_dice_1, "union"),
        ray.CSG(dice, four_dice_2, "union"),
        ray.CSG(dice, four_dice_3, "union"),
        # ray.CSG(dice, four_dice_4, "union"),
        # ray.CSG(dice, four_dice_5, "union"),
        # ray.CSG(dice, three_dice_1, "union"),
        ray.CSG(dice, three_dice_2, "union"),
        ray.CSG(dice, three_dice_3, "union"),
        ray.CSG(dice, three_dice_4, "union"),
        ray.CSG(dice, three_dice_5, "union"),
        # ray.Sphere(vec([0, 0, 0.5]), 0.08, blue)
    ], vec([0,0,0]))

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(.05),
    ]
    
    # camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9)
    # camera = ray.Camera(vec([0, 1.2, 5]), target=vec([0, -.2, 0]), vfov=24, aspect=16 / 9)
    camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -.2, 0]), vfov=24, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def final():
    importlib.reload(ray)
    tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
    # white = ray.Material(vec([1.0, 0.88, 0.65]), k_m=0.5)
    white = ray.Material(vec([1.3, 0.88, 0.65]), k_s=0.3, p=90, k_m=0.8)
    green = ray.Material(vec([0.0, .8, 0.0]), k_s=0.3, p=90, k_m=0.3)

    # sphere = ray.Sphere(vec([0, 0, 0]), 0.66, white)
    sphere = ray.Sphere(vec([0, 0, 0]), 0.70, white)
    # sphere = ray.Sphere(vec([0, 0, 0]), 0.78, white)
    cube = ray.Box(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), white)
    texture = ray.Material(vec([0.2, 0.2, 0.2]), texture="./ref/squares.png")
    # csg_intersection = CSG(sphere, cube, "intersect")

    five_dice_1 = ray.Sphere(vec([0.15, 0.15, 0.5]), 0.075, blue)
    five_dice_2 = ray.Sphere(vec([0, 0, 0.5]), 0.075, blue)
    five_dice_3 = ray.Sphere(vec([-0.15, -0.15, 0.5]), 0.075, blue)
    five_dice_4 = ray.Sphere(vec([0.15, -0.15, 0.5]), 0.075, blue)
    five_dice_5 = ray.Sphere(vec([-0.15, 0.15, 0.5]), 0.075, blue)

    four_dice_1 = ray.Sphere(vec([0.5, -0.15, -0.15]), 0.075, blue)
    four_dice_2 = ray.Sphere(vec([0.5, 0, 0]), 0.075, blue)
    four_dice_3 = ray.Sphere(vec([0.5, 0.15, 0.15]), 0.075, blue)
    four_dice_4 = ray.Sphere(vec([0.5, -0.15, 0.15]), 0.075, blue)
    four_dice_5 = ray.Sphere(vec([0.5, 0.15, -0.15]), 0.075, blue)

    three_dice_1 = ray.Sphere(vec([0, 0.5, 0]), 0.075, blue)
    three_dice_2 = ray.Sphere(vec([0.15, 0.5, -0.15]), 0.075, blue)
    three_dice_3 = ray.Sphere(vec([-0.15, 0.5, 0.15]), 0.075, blue)
    three_dice_4 = ray.Sphere(vec([-0.15, 0.5, -0.15]), 0.075, blue)
    three_dice_5 = ray.Sphere(vec([0.15, 0.5, 0.15]), 0.075, blue)

    floor_sphere_1 = ray.Sphere(vec([1.5, -0.5, -0.5]), 0.2, texture)

    another_cube = ray.Box(vec([-0.5, -0.5, -0.5]), vec([.5, .5, .5]), white)

    dice = ray.CSG(sphere, cube, "intersect")

    rectangle1 = ray.Box(vec([1,0,-1]), vec([2,1,-0.2]), white)
    
    scene = ray.Scene([
        # floor_sphere_1,
        dice,
        # rectangle1, 
        ray.Sphere(vec([0, -40, 0]), 39.5, green),
        # ray.CSG(ray.Sphere(vec([0, -40, 0]), 39.5, green), floor_sphere_1, "subtract"),
        ray.CSG(dice, five_dice_1, "union"),
        ray.CSG(dice, five_dice_2, "union"),
        ray.CSG(dice, five_dice_3, "union"),
        ray.CSG(dice, five_dice_4, "union"),
        ray.CSG(dice, five_dice_5, "union"),
        ray.CSG(dice, four_dice_1, "union"),
        ray.CSG(dice, four_dice_2, "union"),
        ray.CSG(dice, four_dice_3, "union"),
        # ray.CSG(dice, four_dice_4, "union"),
        # ray.CSG(dice, four_dice_5, "union"),
        # ray.CSG(dice, three_dice_1, "union"),
        ray.CSG(dice, three_dice_2, "union"),
        ray.CSG(dice, three_dice_3, "union"),
        ray.CSG(dice, three_dice_4, "union"),
        ray.CSG(dice, three_dice_5, "union"),
        # ray.Sphere(vec([0, 0, 0.5]), 0.08, blue)
    ], vec([0,0,0]))

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(.05),
    ]
    
    # camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9)
    # camera = ray.Camera(vec([0, 1.2, 5]), target=vec([0, -.2, 0]), vfov=24, aspect=16 / 9)
    camera = ray.Camera(vec([3, 1.2, 5]), target=vec([0, -.2, 0]), vfov=24, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
    

def CubeExample():
    importlib.reload(ray)
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
    vs_list = 0.5 * read_obj_triangles(open("cube.obj"))

    scene = ray.Scene([
                      # Make a big sphere for the floor
                      ray.Sphere(vec([0, -40, 0]), 39.5, gray),
                  ] + [
                      # Make triangle objects from the vertex coordinates
                      ray.Triangle(vs, tan) for vs in vs_list
                  ])

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);

def OrthoFriendlyExample(sphere_radius = 0.25):
    gray = ray.Material(vec([0.5, 0.5, 0.5]))

    # One small sphere centered at z=-0.5
    scene = ray.Scene([
        ray.Sphere(vec([0, 0, -0.5]), sphere_radius, gray),
    ])

    lights = [
        ray.AmbientLight(0.5),
    ]
    camera = ray.Camera(vec([0,0,0]), target=vec([0, 0, -0.5]), vfov=90, aspect=1)
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights);
