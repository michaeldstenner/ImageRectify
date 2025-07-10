#!/usr/bin/env python3

import os
import os.path
import sys
import subprocess
import argparse
import re
import json

import numpy as np
from numpy.linalg import inv
import cv2

DEBUG=False
from pprint import pprint

# TODO:
#  - add "verified" checkbox to (PNG/PDF)
#  - verify that detection corners are at code corners (not border corners) and
#    that region definition (with borders added) yield correct
#    dimensions (PNG/PDF)

###############################################################################
## Logging and Debugging functions
###############################################################################
def iprint(*args):
    """print multiple lines indented so they show up nicely in
    FreeCAD's report view"""
    if not DEBUG: return
    s = args[0]
    for v in args[1:]:
        for l in str(v).splitlines():
            s += '\n' + ' '*4 + l
    print(s)

def dbshow(im, points=None):
    """show an image, optionally highlighting specific points in the
    image. points are specified in image/pixel coordinates"""
    if not DEBUG: return
    imc = im.copy()
    if points is not None:
        if type(points) in (list, tuple):
            points = np.array(points).T
        ip = points.astype(int)
        for i in range(ip.shape[1]):
        #print((ip[0,i], ip[1,i]))
            imc = cv2.circle(imc, (ip[0,i], ip[1,i]), 80, (255, 0, 0), 10)
        #marked = cv2.circle(marked, (10, 10), 10, (0, 0, 255), 3)
    cv2.imshow("", imc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################################################
## Calibrated Sheet management - creation and detection
###############################################################################
class CalSheet:
    """Calibrated Sheets - print sheets that are auto-detected with
    rectification and scale.

    Create:
      CalSheet((200,100)).export() # creates a pdf - requires img2pdf
    Detect:
      corners = CalSheet().detect()
      # returns: ((id0, (row0, col0)), ... (id3, (row3, col3)))
    ids:
        0  - top left     - value == 0
        1  - bottom left  - value == sheet height in mm
        2  - bottto right - value == mode number
        3  - top right    - value == sheet width in mm

    mode is future-proofing.  Right now, only "mode 1" is defined, and
    that's what is described above.  Future modes could change
    behavior in some way.  For example, to allow more points,
    different units (imperial), or to allow some multiplier to make
    huge (or tiny) sheets. Mode 1 allows heights/widths ranging from
    12 mm to 1000 mm in 1 mm steps.
    """
    _tagset = (5, 1000)
    def __init__(self, tagset=None):
        t = self.tagset = tagset or self._tagset
        setname = 'DICT_%dX%d_%d' % (t[0], t[0], t[1])
        self.adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, setname))

        #self.size = None if size is None else [ round(s) for s in size ]
        #self.marker_size = marker_size # mm
        #self.font_height = 6 # mm
        #self.label = True
        #self.ppi = 300  # pixers per inch

        #self.ids = None if size is None else \
        #    (0, self.size[0], mode, self.size[1])

    def create(self, size=(185,250), marker_size=14, fmt='svg',
               mode=1, output='.', **kwargs):
        # process size input
        size = parse_geometry(size, cast=int)

        # lets figure out where to write the output
        if os.path.isdir(output):
            filename = f'CalSheet-{size[0]}x{size[1]}.{fmt}'
            path = os.path.join(output, filename)
        elif output.lower().endswith('.' + fmt.lower()):
            path = output
        else:
            path = output + '.' + fmt
        kwargs['path'] = path
        kwargs['ids'] = (0, size[0], mode, size[1])
        if   fmt == 'svg':  return self._create_svg(size,  marker_size, **kwargs)
        elif fmt == 'html': return self._create_html(size, marker_size, **kwargs)
        elif fmt == 'png':  return self._create_png(size,  marker_size, **kwargs)
        elif fmt == 'pdf':  return self._create_pdf(size,  marker_size, **kwargs)
        else: raise Exception(f'unknown format: {fmt}')

    def detect(self, image):
        if type(image) is str: image = load_image(image)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejectedImgPoints = \
            cv2.aruco.detectMarkers(image, self.adict, parameters=parameters)
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        tag_corners = []
        for i, c in zip(ids, corners):
            c1 = tuple([int(k) for k in c[0,0]]) # outer corners
            c3 = tuple([int(k) for k in c[0,2]]) # inner corners
            tag_corners.append( (int(i[0]), c1, c3) )
            #image = cv2.circle(image, (c1[0], c1[1]), 20, (255, 0, 0), 5)
        tag_corners = self._order_corners(tag_corners)
        #dbshow(image, [a[1] for a in tag_corners])
        return tag_corners

    def rectify(self, image, crop=None):
        if type(image) is str: image = load_image(image)
        corners = self.detect(image)
        h = float(corners[1][0])
        w = float(corners[3][0])
        ipoints = np.array([c[1] for c in corners]).T # input points
        tpoints = np.array([c[2] for c in corners]).T # inside corners
        opoints = np.array([[-w/2, h/2], [-w/2, -h/2],
                            [w/2, -h/2], [w/2, h/2]]).T # output points
        A = find_homography(ipoints, opoints)
        A1, A2, size = decompose_homography(image.shape[:2], A)
        rect_image = warp(image, A1, size)
        sheet_points = renormalize(np.matmul(A1, renormalize(ipoints))).astype(int)
        inside_points = renormalize(np.matmul(A1, renormalize(tpoints))).astype(int)
        pitch = A2[0,0]
        abs_diff = abs(np.array(inside_points - sheet_points))
        msize = pitch * sum(sum(abs_diff))/ 8.0
        iprint(f'pitch = {pitch}, msize = {msize}')
        #dbshow(rect_image, sheet_points)
        iprint('-----------------------------------------')
        cimage, shift = self._crop(crop, rect_image, sheet_points,
                                   pitch, msize)
        shiftmm = [s*pitch for s in shift]
        iprint(f'A2 = {A2}')
        iprint(f'cimage.shape = {cimage.shape}')
        iprint(f'shift(pix) = {shift}')
        iprint(f'shift(mm)  = {shiftmm}')
        A2[0,2] += shiftmm[1]
        A2[1,2] -= shiftmm[0]
        iprint(f'A2 = {A2}')
        dbshow(cimage)

        # prepare return
        pitch = float(pitch)
        size = list(reversed([pitch * d for d in cimage.shape[0:2]]))
        transform = dict(xs=A2[0,0], xt=A2[0,2],
                         ys=A2[1,1], yt=A2[1,2])
        transform = { k:float(v) for k,v in transform.items() }
        transform['function'] = \
                  'x_mm, y_mm = x * xs + xt, y_mm = y * ys + yt'
        data = dict(A=A2.tolist(), pitch=pitch, shape=cimage.shape[0:2],
                    size_mm=size, transform=transform)

        return cimage, data

    def _crop(self, crop, image, sheet_p, pitch, msize):
        crop = parse_geometry(crop, ('none', 'inside'), cast=float)
        if crop == 'none':
            return image
        elif crop == 'inside':
            ch = cv = msize # crop from inside corners of markers
        else:
            ch, cv = crop
        cph = int(ch / pitch)
        cpv = int(cv / pitch)
        x,y = sheet_p[0,:], sheet_p[1,:]
        y1, y2 = np.min(y)+cpv, np.max(y)-cpv
        x1, x2 = np.min(x)+cph, np.max(x)-cph
        y1, y2 = max(y1, 0), min(y2, image.shape[0])
        x1, x2 = max(x1, 0), min(x2, image.shape[1])
        return image[y1:y2,x1:x2], (y1, x1)

    ###########################################################################
    # Creation
    ###########################################################################
    def marker(self, id, ppb=1):
        marker_pix = ppb * (self.tagset[0] + 2)
        #print('marker_pix = ', marker_pix)
        marker_image = np.zeros((marker_pix, marker_pix, 1), dtype="uint8")
        # Generate the ArUco marker
        cv2.aruco.generateImageMarker(self.adict, id, marker_pix, marker_image, 1)
        return marker_image

    ##########################################################################
    #  PNG
    def _create_png(self, *args, **kwargs):
        img = self._create_img(*args, **kwargs)
        #output_size = img.shape[0:2]
        o = kwargs['path']
        cv2.imwrite(o, img)

    ##########################################################################
    #  PDF
    def _create_pdf(self, size, *args, **kwargs):
        if 'ppi' not in kwargs: kwargs['ppi'] = 300
        ppi = kwargs['ppi']
        img = self._create_img(size, *args, **kwargs)
        o = kwargs['path']
        pngfile = o[:-4]+'.png'
        cv2.imwrite(pngfile, img)
        ppmm = float(ppi)/25.4
        s = img.shape[0:2]


        #print('Exporting to %s.\nTotal image dimensions are: \n' \
        #      '  %4.1f mm x %4.1f mm\n  %2.3f in x %2.3f in' % \
        #      (outfn, s[1]/ppmm, s[0]/ppmm,
        #       s[1]/ppi, s[0]/ppi))

        cmd = f'img2pdf {pngfile} --imgsize {ppi}dpi --out {o}'
        os.system(cmd)

    ###########################################################################
    #  Raw Pixels
    def _create_img(self, size, marker_size, ppi=300, label=True,
                    verified=True, font_height=None, **kwargs):
        ppmm = ppi / 25.4 # pixels per mm
        # marker pixels-per-bit
        ppb = round(ppmm * marker_size / (self.tagset[0]+2))
        r = self._make_pixel_region(size, ppmm, ppb)
        rs = r.shape

        ids = kwargs['ids']
        o = ppb # outside edge
        for corner in range(4):
            # starting in top left and working counter-clockwise
            signy, signx = [(1, 1), (1, -1), (-1, -1), (-1, 1)][corner]

            m = self.marker(ids[corner], ppb)
            m = np.rot90(m, -corner)

            i = o + m.shape[0] # inside edge
            oy, ox = o * signy, o * signx
            iy, ix = i * signy, i * signx
            sy, ey = min(oy, iy), max(oy, iy) # start/end of slide
            sx, ex = min(ox, ix), max(ox, ix)
            #print(sy, ey, sx, ex)
            r[sy:ey, sx:ex] = m

        if label:
            if font_height is None: font_height = marker_size
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = font_height * ppmm / 21
            color = 0
            thickness = round(fontScale)
            text = '%d mm x %d mm' % (size[0], size[1])
            text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
            while text_size[0] > (size[0]*ppmm - \
                                  4 * ppb * self.tagset[0]):
                #print(text_size)
                fontScale = 0.9 * fontScale
                text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
            x = (r.shape[1] - text_size[0]) // 2
            y = 2*o + text_size[1]
            cv2.putText(r, text, (x,y), font,
                        fontScale, color, thickness, cv2.LINE_AA)
        return r

    def _make_pixel_region(self, size, ppmm, ppb):
        # create the calibrated region
        region_pix = [ round(ppmm * d) for d in size ]
        # pad out the black border row
        region_pix = [ p + 2 * ppb for p in region_pix ]
        #print('region_pix = ', region_pix)
        region_image = np.ones((region_pix[1], region_pix[0], 1), dtype="uint8")*255
        return region_image

    ###########################################################################
    #  HTML
    def _create_html(self, size, marker_size, ids, *args, **kwargs):
        svg = self._create_svg(size, marker_size, ids,
                               *args, **kwargs)
        indented_svg = ''.join([f'  {line}\n' for line in svg.splitlines() ])
        html = '<div style="display: flex; justify-content: '\
            'center; align-items: center; height: 100vh;">\n'
        html += indented_svg
        html += '</div>\n'
        with open(kwargs['path'], 'w') as fo:
                fo.write(svg)

    ###########################################################################
    #  SVG
    def _create_svg(self, size, marker_size, ids, *args, **kwargs):
        xs, ys = size
        svg = '<svg xmlns="http://www.w3.org/2000/svg" ' + \
            f'width="{size[0]}mm" height="{size[1]}mm" ' + \
            f'viewBox="0 0 {size[0]} {size[1]}">\n'
        svg += self._svg_text(size, marker_size, (size[0]/2, marker_size/2),
                                   f"{xs} mm x {ys} mm")
        svg += self._svg_text(size, marker_size, (size[0]/2, size[1]-marker_size/2),
                                   f"(&#9633; verified)")
        trans = [(0, 0), (xs, 0), (xs, ys), (0, ys)]
        for i, id in enumerate(ids):
            code = self._svg_code(id, marker_size)
            tx, ty = trans[i]
            tcode = self._svg_transform(code, i*90, tx, ty)
            svg += tcode
        svg += '</svg>\n'
        if kwargs['path'].endswith('.svg'):
            with open(kwargs['path'], 'w') as fo:
                fo.write(svg)
        return svg

    def _svg_text(self, size, marker_size, center, text):
        xs, ys = size
        vmax = marker_size
        hmax = 2 * (size[0] - 2 * marker_size) / (len(text)+2)
        font_size=min(vmax, hmax)
        xc, yc = center
        tcode = f"""<text x="{xc}" y="{yc}" text-anchor="middle" dominant-baseline="middle"
                    font-size="{font_size}">\n  {text}\n</text>\n"""
        return tcode

    def _svg_transform(self, svg, rot, tx, ty):
        indented_svg = ''.join([f'  {line}\n' for line in svg.splitlines() ])
        tsvg = f'<g transform="translate({tx}, {ty}) rotate({rot})">\n' + \
            indented_svg + \
            '</g>\n'
        return tsvg

    def _svg_code(self, id, marker_size):
        img = self.marker(id)
        h, w = img.shape[0:2]
        p = marker_size/h
        svg = ''
        # Generate rectangles for each pixel
        for y in range(h):
            for x in range(w):
                if not img[y, x]:
                    svg += f'<rect x="{x*p}" y="{y*p}" width="{p}" height="{p}" />\n'
        return svg

    ###########################################################################
    # Detection
    ###########################################################################
    def _order_corners(self, corners):
        """Identify and order the corners
        OpenCV tells us which tags it found, but we don't know the proper arrangement.
        This puts them in the proper order.
        """

        if not len(corners) == 4:
            raise Exception('expect 4 corners: found %d' % len(corners))
        sc = []
        for c in corners:
            if c[0] == 0:      # 0 is always in the top left
                top_left = c
                continue
            if c[0] < 13:      # the "mode" indicator (1-12) is in the bottom right
                bottom_right = c
                continue
            sc.append(c)

        # we know the TL/BR.  Only two arrangments remain.  We'll choose the one that
        # gives us negative area - that is - that sends us around counter-clockwise
        oc = [top_left, sc[0], bottom_right, sc[1]]
        coords = [ c[1] for c in oc ]
        area = self._polygon_area(coords)
        if area < 0:
            return oc
        else:
            return [top_left, sc[1], bottom_right, sc[0]]

    def _polygon_area(self, polygon):
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        return 0.5 * area


##############################################################################
## Image Warping and Homography Estimation
##############################################################################
def warp(image, A, size):
    """apply the transformation matrix A to recitfy the image,
    creating an image of size "size"."""
    rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rect_image = cv2.warpPerspective(rgba, A, size,
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_TRANSPARENT)
    #dbshow(rect_image)
    return rect_image

def find_homography(points_source, points_target):
    """find the homography (transformation matrix) which maps source
    coordinates to target coordinates"""
    A  = _construct_A(points_source[0:2,:].T, points_target[0:2,:].T)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # Solution to H is the last column of V, or last row of V transpose
    homography = vh[-1].reshape((3,3))
    return homography/homography[2,2]

def _construct_A(points_source, points_target):
    assert points_source.shape == points_target.shape, "Shape does not match"
    num_points = points_source.shape[0]
    #print('num points = ', num_points)
    matrices = []
    for i in range(num_points):
        partial_A = _construct_A_partial(points_source[i,:], points_target[i,:])
        matrices.append(partial_A)
    return np.concatenate(matrices, axis=0)

def _construct_A_partial(point_source, point_target):
    #print(point_target)
    x, y, z = point_source[0], point_source[1], 1
    x_t, y_t, z_t = point_target[0], point_target[1], 1

    A_partial = np.array([
        [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
        [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
    ])
    return A_partial

def decompose_homography(shape, A):
    """Decompose homography A into A2*A1 such that, when applied
    to an image of size 'shape':
       - A1 transforms the image file in image coordinates with
         reasonable pixel density and fully contains the pixels.
         This represents the physical image-to-image transform.
       - A2 simply scales coordinates (from 1=pixel to 1=mm) and
         translates.  This represents the coordinate change from
         transformed image to real-world coordinates.
    This also return "size", which is the target dimensions of the
    transformed image which satisfies the conditions described above.
    """
    s = (shape[0]-1, shape[1]-1)
    c = np.matrix( [[0, 0,    s[1], s[1]],      # input corner pixel centers
                    [0, s[0], s[0], 0],
                    [1, 1,    1,    1]] )
    oc = np.matmul(A, c)
    oc = renormalize(oc) # output sketch corners
    iprint('input image corners: c = ', c)
    iprint('output world corners: A2A1 c = ', oc)
    tx = oc[0,:].min()       # x translation
    ty = oc[1,:].min()       # y translation
    lx = oc[0,:].max() - tx  # x width
    ly = oc[1,:].max() - ty  # y height

    P = s[0] * s[1] * 8 # total pixels
    ar = lx/ly      # output aspect ratio
    iprint('output aspect ratio: ', ar)
    xp = np.sqrt(P*ar)
    yp = np.sqrt(P/ar)
    s = lx / xp  # scale
    T = np.matrix( [[1, 0, tx], [0, 1, +ly+ty], [0, 0, 1]] )
    iprint('translate: ', T)
    S = np.matrix( [[s, 0, 0],  [0, -s, 0],     [0, 0, 1]] )
    iprint('scale: ', S)
    A2 = np.matmul(T, S)
    A1 = np.matmul(inv(A2), A)
    iprint('input corners: c = ', c)
    iprint('target output image size: %d x %d' % (xp, yp))
    iprint('output corners: A1 c = ', renormalize(np.matmul(A1, c)))

    size = int(xp), int(yp)
    return A1, A2, size

##############################################################################
## Coordinate Manipulation
##############################################################################

def renormalize(v, strip=False):
    """homogenous coordinate renormalization
    (there's probably a cleaner way to do this)"""
    r,c = v.shape[:2]
    if r == 2: # if it doesn't have the homogenous row, add it
        v = np.vstack((v, np.ones((1,c))))
    else: # renormalize the homogenous coordinate to 1
        r3 = np.matrix(v[2,:])
        o = np.ones((3,1))
        d = np.matmul(o, r3)
        v = v / d
    if strip: v = v[:-1] # remove the homogenous row if desired
    return v

##############################################################################
## file stuff
##############################################################################

def load_image(img_path):
    base, ext = os.path.splitext(img_path)
    if ext.lower() == '.heic':
        img_path = convert_heic_to_jpeg(img_path)
    img = cv2.imread(img_path)
    return img

def convert_heic_to_jpeg(heic_path):
    base, ext = os.path.splitext(heic_path)
    jpeg_path = base + '.jpeg'
    cmd = ['magick', heic_path, jpeg_path]

    print('running: ', ' '.join(cmd))
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print(e)
        sys.exit()

    st = cp.stdout.strip()
    if st:
        print('STDOUT:')
        for s in st.split('\n'): print(s)
    st = cp.stderr.strip()
    if st:
        print('STDERR:')
        for s in st.split('\n'): print(s)
    #with Image.open(heic_path) as img:
    #    img.convert('RGB').save(jpeg_path, 'JPEG')
    return jpeg_path

##############################################################################
## Helpers
##############################################################################

def parse_geometry(geom, specials=(), cast=int):
    e = Exception(f'Bad Geometry: {repr(geom)}')
    if geom in specials:
        return geom
    if type(geom) in (tuple, list) and len(geom) == 2:
        for g in geom:
            if type(g) not in (float, int):
                raise(e)
    elif type(geom) is str and re.match(r'^\d+\.?\d*(x\d+\.?\d*)?$', geom):
        geom = geom.split('x')
    elif type(geom) in (float, int):
        geom = (geom, geom)
    else:
        raise e

    if len(geom) == 1: geom = (geom[0], geom[0])
    geom = tuple([cast(g) for g in geom])
    return geom

##############################################################################
## Script Use
##############################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    a = parser.add_argument
    a("-s", "--size", default='185x250',
      help="sheet dimensions to create (default=185x250)")
    a("-S", "--sheet-format", default="html",
      help="Output format: html, svg, png, pdf")
    a("-o", "--output", default=None,
      help="output dir, file basename, or filename")
    a("-c", "--crop", default=0,
      help="mm to crop from markers (+ or -) or 'inside'")
    a("-j", "--json", default=False, action='store_true',
      help="Print data out as json for processed images")
    a("-D", "--debug", default=False, action='store_true',
      help="turn on debugging output")
    a("images", nargs='*',
      help="Image(s) to rectify")

    op = parser.parse_args()
    return op


def main(**kwargs):
    op = parse_args()
    for k, v in kwargs.items():
        setattr(op, k, v)
    if op.debug:
        global DEBUG
        DEBUG = True
    if op.images:
        suffix = '_rect.png'
        d = {}
        for filename in op.images:
            img, data = CalSheet().rectify(filename, op.crop)
            if op.output is None:
                odir, fn = os.path.split(filename)
                ofn = os.path.splitext(fn)[0] + suffix
                opath = os.path.join(odir, ofn)
            elif os.path.isdir(op.output):
                fn = os.path.split(filename)[1]
                odir = op.output
                ofn = os.path.splitext(fn)[0] + suffix
                opath = os.path.join(odir, ofn)
            else:
                opath = os.output
            cv2.imwrite(opath, img)
            data['output'] = opath
            d[filename] = data
            if not op.json: print(f'{opath}: {data["pitch"]:.6f}')
        if op.json: print(json.dumps(d, indent=4))
    else:
        if op.output is None: op.output = '.'
        CalSheet().create(fmt=op.sheet_format,size=op.size,
                          output=op.output)

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    main()
