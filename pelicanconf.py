AUTHOR = 'Anwaar Khalid'
SITENAME = ''
SITEURL = 'http://localhost:8000'
SITE_LOGO = '/assets/images/logo.png'
COPYRIGHT_YEAR = 2026
COPYRIGHT_NAME = AUTHOR

PATH = 'content'
TIMEZONE = 'Asia/Kolkata'
DEFAULT_LANG = 'en'

#SITE STRUCTURE
INDEX_SAVE_AS = 'blog/index.html'
DIRECT_TEMPLATES = ['index', 'tags', 'categories', 'authors']

THEME = "attila-2.0"
HOME_COVER = '/assets/images/cover1.jpg'
HOME_COLOR = 'black'

STATIC_PATHS = ['assets']
RELATIVE_URLS = False

EXTRA_PATH_METADATA = {
    'assets/images/favicon.ico': {'path': 'favicon.ico'},
}

#ARTICLE URLS AND PATHS 
ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{slug}/index.html'

PAGE_URL = 'pages/{slug}/'
PAGE_SAVE_AS = 'pages/{slug}/index.html'

# Tags and Category path
CATEGORY_URL = 'category/{slug}/'
CATEGORY_SAVE_AS = 'category/{slug}/index.html'
CATEGORIES_URL = 'category/'
CATEGORIES_SAVE_AS = 'category/index.html'

TAG_URL = 'tag/{slug}/'
TAG_SAVE_AS = 'tag/{slug}/index.html'
TAGS_URL = 'tag/'
TAGS_SAVE_AS = 'tag/index.html'

# Author
AUTHOR_URL = 'author/{slug}/'
AUTHOR_SAVE_AS = 'author/{slug}/index.html'
AUTHORS_URL = 'author/'
AUTHORS_SAVE_AS = 'author/index.html'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# PAGINATION
DEFAULT_PAGINATION = 10
PAGINATION_PATTERNS = (
    (1, '{base_name}/', '{base_name}/index.html'),
    (2, '{base_name}/page/{number}/', '{base_name}/page/{number}/index.html'),
)

# MARKDOWN CONFIGURATION
MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.codehilite": {"css_class": "highlight"},
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
        "markdown.extensions.smarty": {},
        "markdown.extensions.toc": {
            "title": "Table of Contents",
            "marker": "[TOC]",
            "permalink": True,
        },
    },
    "output_format": "html5",
}

# PLUGINS
PLUGINS = [
    "pelican.plugins.neighbors",
    "pelican.plugins.seo",
    "pelican.plugins.sitemap",
    "pelican.plugins.webassets",
    "pelican.plugins.render_math",
]

# SITEMAP CONFIGURATION
SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.5,
        'indexes': 0.5,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}

# COMMENTS
DISQUS_SITENAME = "anwaar-khalid"

# META CONFIGURATION
#TAG_META = {
#}

CATEGORY_META = {
  "compression": {
    "cover": "https://images.unsplash.com/photo-1645113720391-279a153b4f53?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2073&q=80",
    "description": "compresssssssssssionnneioioi"
  },
}

AUTHOR_META = {
  "anwaar khalid": {
    "name": "Anwaar Khalid",
    "cover": "/assets/images/cover.jpg",
    "image": "/assets/images/logo.png",
    "website": "vive-discere.com",
    "linkedin": "anwaar-khalid",
    "email": "khalidanwaar0@gmail.com",
    "github": "hello-fri-end",
    "location": "Bangalore, Karnataka",
    "bio": "To live is to learn. Hope you find something useful here :)"
  }
}

## Social widget
#SOCIAL = (('LinkedIn', 'https://linkedin.com/in/anwaar-khalid'),
#          ('GitHub', 'https://github.com/hello-fri-end'),
#          )


# THEME DISPLAY
SHOW_ARTICLE_MODIFIED_TIME = False
SHOW_AUTHOR_BIO_IN_ARTICLE = False
SHOW_CATEGORIES_ON_MENU = False
SHOW_COMMENTS_COUNT_IN_ARTICLE_SUMMARY = False
SHOW_CREDITS = False
FOOTER_SOCIAL_ICONS = True
SHOW_FULL_ARTICLE_IN_SUMMARY = False
SHOW_PAGES_ON_MENU = False
SHOW_SITESUBTITLE_IN_HTML_TITLE = False
SHOW_TAGS_IN_ARTICLE_SUMMARY = False

# MENU ITEMS
MENUITEMS = (('Home', '/'),
             ('Blog', '/author/anwaar-khalid/'),
            )

# Jinja config - Pelican 4
JINJA_ENVIRONMENT = {
  'extensions' :[
    'jinja2.ext.loopcontrols',
    'jinja2.ext.i18n',
    'jinja2.ext.do',
  ]
}
