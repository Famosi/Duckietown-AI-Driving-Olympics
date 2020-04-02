# coding=utf-8
import math

from . import logger

import numpy as np



from ctypes import byref

from .utils import *

class Texture(object):
    """
    Manage the caching of textures, and texture randomization
    """

    # List of textures available for a given path
    tex_paths = {}

    # Cache of textures
    tex_cache = {}

    @classmethod
    def get(self, tex_name, rng=None):
        paths = self.tex_paths.get(tex_name, [])

        # Get an inventory of the existing texture files
        if len(paths) == 0:
            for i in range(1, 10):
                path = get_file_path('textures', '%s_%d' % (tex_name, i), 'png')
                if not os.path.exists(path):
                    break
                paths.append(path)

        assert len(paths) > 0, 'failed to load textures for name "%s"' % tex_name

        if rng:
            path_idx = rng.randint(0, len(paths))
            path = paths[path_idx]
        else:
            path = paths[0]

        if path not in self.tex_cache:
            self.tex_cache[path] = Texture(load_texture(path))

        return self.tex_cache[path]

    def __init__(self, tex):
        assert not isinstance(tex, str)
        self.tex = tex

    def bind(self):
        from pyglet import gl
        gl.glBindTexture(self.tex.target, self.tex.id)

def load_texture(tex_path):
    from pyglet import gl
    logger.debug('loading texture "%s"' % os.path.basename(tex_path))
    import pyglet
    img = pyglet.image.load(tex_path)
    tex = img.get_texture()
    gl.glEnable(tex.target)
    gl.glBindTexture(tex.target, tex.id)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        img.width,
        img.height,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        img.get_image_data().get_data('RGBA', img.width * 4)
    )

    return tex

def create_frame_buffers(width, height, num_samples):
    """Create the frame buffer objects"""
    from pyglet import gl

    # Create a frame buffer (rendering target)
    multi_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(multi_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    try:
        # Create a multisampled texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures( 1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex)
        gl.glTexImage2DMultisample(
            gl.GL_TEXTURE_2D_MULTISAMPLE,
            num_samples,
            gl.GL_RGBA32F,
            width,
            height,
            True
        )
        gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D_MULTISAMPLE,
            fbTex,
            0
        )

        # Attach a multisampled depth buffer to the FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, num_samples, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    except:
        logger.debug('Falling back to non-multisampled frame buffer')

        # Create a plain texture texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures( 1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            fbTex,
            0
        )

        # Attach depth buffer to FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    # Sanity check
    import pyglet
    if pyglet.options['debug_gl']:
      res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
      assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    final_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(final_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)

    # Create the texture used to resolve the final render
    fbTex = gl.GLuint(0)
    gl.glGenTextures(1, byref(fbTex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
    gl.glTexImage2D(
        gl. GL_TEXTURE_2D,
        0,
        gl.GL_RGBA,
        width,
        height,
        0,
        gl. GL_RGBA,
        gl.GL_FLOAT,
        None
    )
    gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
        fbTex,
        0
    )
    import pyglet
    if pyglet.options['debug_gl']:
      res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
      assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Enable depth testing
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Unbind the frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return multi_fbo, final_fbo

def rotate_point(px, py, cx, cy, theta):
    """
    Rotate a 2D point around a center
    """

    dx = px - cx
    dy = py - cy

    new_dx = dx * math.cos(theta) + dy * math.sin(theta)
    new_dy = dy * math.cos(theta) - dx * math.sin(theta)

    return cx + new_dx, cy + new_dy

def gen_rot_matrix(axis, angle):
    """
    Rotation matrix for a counterclockwise rotation around the given axis
    """

    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def bezier_point(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p  = ((1-t)**3) * cps[0,:]
    p += 3 * t * ((1-t)**2) * cps[1,:]
    p += 3 * (t**2) * (1-t) * cps[2,:]
    p += (t**3) * cps[3,:]

    return p

def bezier_tangent(cps, t):
    """
    Tangent of a cubic Bezier curve (first order derivative)
    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    """

    p  = 3 * ((1-t)**2) * (cps[1,:] - cps[0,:])
    p += 6 * (1-t) * t * (cps[2,:] - cps[1,:])
    p += 3 * (t ** 2) * (cps[3,:] - cps[2,:])

    norm = np.linalg.norm(p)
    p /= norm

    return p

def bezier_closest(cps, p, t_bot=0, t_top=1, n=8):
    mid = (t_bot + t_top) * 0.5

    if n == 0:
        return mid

    p_bot = bezier_point(cps, t_bot)
    p_top = bezier_point(cps, t_top)

    d_bot = np.linalg.norm(p_bot - p)
    d_top = np.linalg.norm(p_top - p)

    if d_bot < d_top:
        return bezier_closest(cps, p, t_bot, mid, n-1)

    return bezier_closest(cps, p, mid, t_top, n-1)


def bezier_draw(cps, n = 20, red=False, env=None, kind=None):
    from pyglet import gl
    pts = [bezier_point(cps, i/(n-1)) for i in range(0, n)]

    # if env is not None and kind is not None:
    #     if not kind in env.pts:
    #         env.pts[kind] = pts
    if env is not None:
        if env.exclude == 12:
            env.pts.append(pts)
        if env.exclude == 0:
            env.exclude = 13
        env.exclude -= 1

    gl.glBegin(gl.GL_LINE_STRIP)

    if red:
        gl.glColor3f(1, 0, 0)
    else:
        gl.glColor3f(0, 0, 1)

    for i, p in enumerate(pts):
        gl.glVertex3f(*p)

    gl.glEnd()
    gl.glColor3f(1,1,1)

############################################################################################################
############################################################################################################
# @riza

def bezier_draw_points(cps, n=6, red=True, draw=True):
    """
    Draw points on the directory vector line
    :param cps: Control points, (x,y,z) coordinates of start&end points of the line
    :param n: # of points to be shown
    :return: None
    """
    # Draw the points on the dir_vec
    from pyglet import gl
    pts = [get_linear_bezier(cps, t) for t in np.linspace(0, 1, n)]

    if draw:
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        if red:
            gl.glColor3f(0, 0, 1)
        else:
            gl.glColor3f(1, 0, 0)

        for i, p in enumerate(pts):
            gl.glVertex3f(*p)

        gl.glEnd()
        gl.glColor3f(1,1,1)

    return pts



def bezier_draw_points_curve(cps, n = 20, red=False):
    from pyglet import gl
    pts = [bezier_point(cps, i/(n-1)) for i in range(0,n)]
    gl.glPointSize(5)
    gl.glBegin(gl.GL_POINTS)

    if red:
        gl.glColor3f(0, 0, 1)
    else:
        gl.glColor3f(1, 0, 0)

    for i, p in enumerate(pts):
        gl.glVertex3f(p[0], 0.01, p[2])

    gl.glEnd()
    gl.glColor3f(1,1,1)


def bezier_draw_line(cps, grid_coords=None, get_tile=None, n=6, perpendicular=False):
    """
    Draw directory vector line
    :param draw_perpendicular:
    :param n: # of points to be shown
    :param cps: (x,y,z) coordinates of start&end points of the line
    :return: None
    """
    from pyglet import gl
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(*cps[0])
    gl.glVertex3f(*cps[1])
    gl.glEnd()

    _sum = 0

    if perpendicular:
        pts = [get_linear_bezier(cps, t) for t in np.linspace(0, 1, n)]
        # (y2-y1)/(x2-x1)
        slope = (cps[0][2] - cps[1][2]) / (cps[0][0] - cps[1][0])

        for p in pts:
            # k = y + x/slope
            k = p[2] + p[0]/slope
            p_1 = np.array([0, 0.01, k])
            # get unit dir_vec of perpendicular line
            x_, _, y_ = (p_1 - p)
            norm = np.linalg.norm([x_, y_])
            x_ /= norm
            y_ /= norm

            dir_start = [p[0] + 0.2 * x_, 0.01, p[2] + 0.2 * y_]
            dir_end   = [p[0] - 0.2 * x_, 0.01, p[2] - 0.2 * y_]

            # bezier_draw_line(np.vstack((dir_start, dir_end)))
            # TODO: INCREASE n PARAMETER IN bezier_draw_points
            points = bezier_draw_points(np.vstack((dir_start, dir_end)), red=True, draw=False, n=12)

            prev_dist = 10
            pt_line = [0]*3
            pt_curve = [0]*3

            k, m = grid_coords(p)

            curves = get_tile(k, m)['curves']


            for i in points:
                # Find closest point and tangent to this curve
                # TODO: INCREASE n PARAMETER IN bezier_closest
                t = bezier_closest(curves[1], i, n=12)
                point = bezier_point(curves[1], t)

                dist = np.linalg.norm(i-point)
                if dist < prev_dist:
                    prev_dist = dist
                    pt_line = i
                    pt_curve = point

            draw_intersection_point_curve(pt_curve)
            draw_intersection_point_line(pt_line)

            # Get distance from intersection point to directory vector line
            # print("\n", np.linalg.norm(pt_curve - p))
            _sum += np.linalg.norm(pt_curve - p)
            #
            # print(sum, np.linalg.norm(pt_curve - p))
            # print("SUM", sum)

            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1, 1, 1)
            gl.glVertex3f(pt_curve[0], 0.01, pt_curve[2])
            gl.glVertex3f(p[0], 0.01, p[2])
            gl.glEnd()

    return _sum

        # print("\n\n")


def draw_intersection_point_curve(point):
    """
    Draw intersection points one-by-one b/w bezier curve and perpendicular lines
    :param point: Coordinates of the point
    """
    from pyglet import gl
    gl.glPointSize(7)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(1, 1, 1)
    gl.glVertex3f(point[0], 0.1, point[2])
    gl.glEnd()

def draw_intersection_point_line(point):
    """
    :param point: Coordinates of points on perpendicular line that is the closest to intersection point
    """
    from pyglet import gl
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(point[0], 0.1, point[2])
    gl.glEnd()

def get_linear_bezier(cps, t):
    """
    Linear Bezier curve interpolation
    B(t) = (1-t)*P0 + t*P1
    """
    p = (1-t) * cps[0, :]
    p += t * cps[1, :]
    return p


