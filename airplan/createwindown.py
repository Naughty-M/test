import pygame
from pygame.locals import *


class Heroplan(object):
    def __init__(self, screen):
        self.X = 50
        self.Y = 50
        self.screen = screen
        self.imageName = "./file/hero1.png"
        self.image = pygame.image.load(self.imageName)
        # ⽤来存储英雄⻜机发射的所有⼦弹
        self.bulletList = []

    def display(self):
        self.screen.blit(self.image, (self.X, self.Y))

        for bullet in self.bulletList:
            bullet.display()
            bullet.move()

    def moveLeft(self): self.X -= 10

    def moveRight(self): self.X += 10

    def sheBullet(self):
        newBullet = Bullet(self.X, self.Y, self.screen)
        self.bulletList.append(newBullet)


def key_control(heroPlane):
    # 判断是否是点击了退出按钮
    for event in pygame.event.get():
        print(event.type)
        if event.type == QUIT:
            print("exit")
            exit()
        elif event.type == KEYDOWN:
            if event.key == K_a or event.key == K_LEFT:
                print('left')
                heroPlane.moveLeft()
            elif event.key == K_d or event.key == K_RIGHT:
                print('right')
                heroPlane.moveRight()
            elif event.key == K_SPACE:
                heroPlane.sheBullet()


class Bullet(object, ):
    def __init__(self, x, y, screen):
        self.X = x
        self.Y = y
        self.screen = screen
        self.image = pygame.image.load("./file/bullet.png")

    def move(self):
        self.Y -= 5

    def display(self):
        self.screen.blit(self.image, (self.X, self.Y))


class EnemyPlane(object):
    def __init__(self, screen):
        # 设置⻜机默认的位置
        self.x = 0
        self.y = 0
        # 设置要显示内容的窗⼝

        self.screen = screen
        self.imageName = "./file/enemy0.png"
        self.image = pygame.image.load(self.imageName)

    def display(self):
        self.screen.blit(self.image, (self.x, self.y))
        self.move()
    def move(self):
        self.x+=6
        self.y+=3

def main():
    screen = pygame.display.set_mode((480, 600), 0, 32)
    background = pygame.image.load('./file/background.png')
    heroPlane = Heroplan(screen)
    enemyPlane = EnemyPlane(screen)

    while True:
        screen.blit(background, (0, 0))
        heroPlane.display()
        enemyPlane.display()
        key_control(heroPlane)
        pygame.display.update()


if __name__ == '__main__':
    main()
