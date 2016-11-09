from bs4 import BeautifulSoup
import urllib2
import logging
import re
import os


class PianoMidiDeDownloader(object):
    resource_name = "Piano-midi.De "

    def __init__(self):
        super(PianoMidiDeDownloader, self).__init__()

    def get_resource_name(self):
        return self.resource_name

    def __fetch_url(self, url):
        request = urllib2.Request(url)
        return urllib2.urlopen(request)

    def parse_midi_urls(self):
        base_url = 'http://www.piano-midi.de'
        response = self.__fetch_url(base_url + '/midi_files.htm')
        soup = BeautifulSoup(response, "html.parser")
        composers = {}
        for cell in soup.findAll("td", {"class": "midi"}):
            link_nodes = cell.findChildren('a')
            comp_node = link_nodes[0] if link_nodes else None
            if comp_node is None:
                continue
            comp_name = re.sub("\s\s*", " ", comp_node.get_text())
            composers[base_url + '/' + comp_node['href']] = comp_name
        urls = []
        for comp_url, comp_name in composers.iteritems():
            response = self.__fetch_url(comp_url)
            soup = BeautifulSoup(response, "html.parser")
            num_midi_files = 0
            for midi_node in soup.findAll("a", {"class": "navi"}):
                midi_url = midi_node['href']
                if not midi_url.lower().endswith('.mid'):
                    continue
                urls.append(base_url + '/' + midi_url)
                num_midi_files += 1
            print "Composer: %s \t %d files" % (comp_name, num_midi_files)
        return urls

    def download_urls(self, urls):
        for i in xrange(len(urls)):
            u = urls[i]
            fname = u.split('/')[-1]
            comp = u.split('/')[-2]
            content = self.__fetch_url(u).read()
            p = os.path.join('music-db/piano-midi.de', comp)
            if not os.path.isdir(p):
                os.makedirs(p)
            with open(os.path.join(p, fname.lower()), 'w') as f:
                f.write(content)
                print "%d/%d %s" % (i + 1, len(urls), fname)


if __name__ == '__main__':
    downloader = PianoMidiDeDownloader()
    urls = downloader.parse_midi_urls()
    print "Downloading %d files.." % len(urls)
    downloader.download_urls(urls)
